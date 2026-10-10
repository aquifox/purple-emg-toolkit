"""RAW 1 kHz two-stage RF/CNN real-time classifier (no recording)."""

from __future__ import annotations

import queue
import os
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import serial
from serial.tools import list_ports


SOURCE_DIR = Path(__file__).resolve().parent
AI_SOURCE_DIR = SOURCE_DIR.parent / "EMG_AI"
if AI_SOURCE_DIR.is_dir() and str(AI_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SOURCE_DIR))

from raw_two_stage_common import (  # noqa: E402
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    NOTCH_FREQUENCIES_HZ,
    NOTCH_HALF_WIDTH_HZ,
    SAMPLE_RATE_HZ,
    WINDOW_SAMPLES,
    channel_rms,
    extract_rf_features,
    normalize_for_classifier,
    preprocess_window,
)
from PurpleTeam_EMG_RAW_Recorder import (  # noqa: E402
    BAUD_RATE,
    BG,
    CHANNEL_COLORS,
    DANGER,
    GRID,
    MUTED,
    PANEL,
    PANEL_2,
    SUCCESS,
    TEXT,
    WARNING,
    RawPacketParser,
    project_directory,
)


AI_MODES = ("Random Forest", "1D CNN")
REST_LABEL = "휴식"
UNKNOWN_LABEL = "판별 불가"
GRAPH_SAMPLES = 2_000
PREDICTION_HOP = 100
BASELINE_WINDOW_COUNT = 21
ACTIVATION_GATE_SAMPLES = 500
ACTIVATION_OFF_CONFIRMATIONS = 3
GESTURE_SWITCH_CONFIRMATIONS = 4
GESTURE_SWITCH_MARGIN = 0.10


def model_output_directory() -> Path:
    """Select a model set without ever loading both sets into memory."""
    requested = os.environ.get("PURPLE_EMG_MODEL_SET", "").strip()
    if requested:
        output_name = requested
    else:
        executable_name = (
            Path(sys.executable).stem if getattr(sys, "frozen", False) else Path(__file__).stem
        )
        output_name = (
            "outputs_all_raw_dates_5class"
            if "전체날짜" in executable_name
            else "outputs_final_5class"
        )
    return project_directory() / "EMG_AI" / output_name


class RawEmgAiClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("퍼플팀 RAW EMG 단일 모델 AI 판별기")
        self.root.configure(bg=BG)
        width = min(1220, max(1020, root.winfo_screenwidth() - 80))
        height = min(900, max(760, root.winfo_screenheight() - 80))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1000, 720)

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.reader_stop = threading.Event()
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.parser = RawPacketParser()
        self.port_map: dict[str, str] = {}
        self.last_packet_at = 0.0
        self.last_sample_index: int | None = None
        self.total_samples = 0
        self.total_gaps = 0
        self.samples_since_prediction = 0

        self.raw_buffer: deque[tuple[float, float, float]] = deque(maxlen=3_000)
        self.history: deque[tuple[int, int, int, int]] = deque(maxlen=GRAPH_SAMPLES)
        self.latest_values = (0, 0, 0)

        self.rf_model: object | None = None
        self.cnn_model: object | None = None
        self.torch_module: object | None = None
        self.loaded_mode: str | None = None
        self.metadata: dict[str, object] = {}
        self.gesture_labels: list[str] = []
        self.model_baseline_rms = np.ones(3, dtype=np.float32)
        self.live_baseline_rms: np.ndarray | None = None
        self.activation_on = 4.0
        self.activation_off = 2.5
        self.rf_confidence_threshold = 0.65
        self.cnn_confidence_threshold = 0.65
        self.cnn_scale = 1.0
        self.filter_low_hz = FILTER_LOW_HZ
        self.filter_high_hz = FILTER_HIGH_HZ
        self.notch_frequencies_hz = tuple(NOTCH_FREQUENCIES_HZ)
        self.notch_half_width_hz = NOTCH_HALF_WIDTH_HZ
        self.classification_normalization = "none"
        self.rf_feature_version = "legacy_v1"

        self.calibrating_baseline = False
        self.baseline_measurements: list[np.ndarray] = []
        self.activation_score = 0.0
        self.latest_rms = np.zeros(3, dtype=np.float32)
        self.active_state = False
        self.on_counter = 0
        self.off_counter = 0
        self.stable_gesture_label: str | None = None
        self.switch_candidate: str | None = None
        self.switch_counter = 0

        self.rf_probability_ema: np.ndarray | None = None
        self.cnn_probability_ema: np.ndarray | None = None
        self.latest_rf_prediction = "꺼짐"
        self.latest_cnn_prediction = "꺼짐"
        self.latest_rf_confidence = 0.0
        self.latest_cnn_confidence = 0.0

        self._configure_styles()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.refresh_ports()
        self.root.after(30, self.process_messages)
        self.root.after(100, self.refresh_display)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.TCombobox",
            fieldbackground=PANEL_2,
            background=PANEL_2,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor="#3b4560",
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", PANEL_2)],
            foreground=[("readonly", TEXT)],
        )

    @staticmethod
    def _button(
        parent: tk.Widget, text: str, command, color: str, width: int = 10
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=10,
            pady=9,
            width=width,
            font=("Malgun Gothic", 9, "bold"),
            cursor="hand2",
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 12))
        tk.Label(
            header,
            text="퍼플팀 RAW EMG 단일 모델 AI 판별기",
            bg=BG,
            fg=TEXT,
            font=("Malgun Gothic", 20, "bold"),
        ).pack(side="left")
        tk.Label(
            header,
            text="휴식 게이트 → 중지/약지 판정 → 불확실하면 판별 불가",
            bg=BG,
            fg=WARNING,
            font=("Malgun Gothic", 9, "bold"),
        ).pack(side="left", padx=16)
        self.connection_var = tk.StringVar(value="● 연결 안 됨")
        tk.Label(
            header,
            textvariable=self.connection_var,
            bg=BG,
            fg=MUTED,
            font=("Malgun Gothic", 10),
        ).pack(side="right")

        controls = tk.Frame(self.root, bg=PANEL, padx=14, pady=11)
        controls.pack(fill="x", padx=24, pady=(0, 10))
        tk.Label(
            controls, text="포트", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 9, "bold")
        ).pack(side="left", padx=(2, 8))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            controls,
            textvariable=self.port_var,
            state="readonly",
            width=33,
            style="Dark.TCombobox",
        )
        self.port_combo.pack(side="left")
        self._button(controls, "새로고침", self.refresh_ports, "#30384d", 9).pack(
            side="left", padx=7
        )
        self.connect_button = self._button(
            controls, "연결", self.toggle_connection, "#6040a8", 9
        )
        self.connect_button.pack(side="left")
        self._button(
            controls,
            "휴식 기준 다시 설정",
            self.start_baseline_calibration,
            "#7a5b24",
            17,
        ).pack(side="right")

        status = tk.Frame(
            self.root,
            bg=PANEL_2,
            highlightthickness=1,
            highlightbackground="#31394d",
        )
        status.pack(fill="x", padx=24, pady=(0, 10))
        self.status_var = tk.StringVar(
            value="모델 하나를 선택하고 [선택 모델 켜기]를 누르세요"
        )
        self.status_label = tk.Label(
            status,
            textvariable=self.status_var,
            bg=PANEL_2,
            fg=MUTED,
            font=("Malgun Gothic", 10, "bold"),
            anchor="w",
            pady=11,
        )
        self.status_label.pack(fill="x", padx=16)

        ai_panel = tk.Frame(
            self.root,
            bg=PANEL,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground="#31394d",
        )
        ai_panel.pack(fill="x", padx=24, pady=(0, 10))
        ai_header = tk.Frame(ai_panel, bg=PANEL)
        ai_header.pack(fill="x", pady=(0, 8))
        tk.Label(
            ai_header, text="실시간 판정", bg=PANEL, fg=TEXT,
            font=("Malgun Gothic", 11, "bold")
        ).pack(side="left")
        self.ai_mode_var = tk.StringVar(value=AI_MODES[0])
        self.ai_mode_combo = ttk.Combobox(
            ai_header,
            textvariable=self.ai_mode_var,
            values=AI_MODES,
            state="readonly",
            width=15,
            style="Dark.TCombobox",
        )
        self.ai_mode_combo.pack(side="left", padx=10)
        self.ai_mode_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_ai_display()
        )
        self.model_load_button = self._button(
            ai_header, "선택 모델 켜기", self.load_models, "#3b4560", 15
        )
        self.model_load_button.pack(side="left")
        self.model_status_var = tk.StringVar(value="RF 또는 CNN 하나를 선택해 켜세요")
        tk.Label(
            ai_header,
            textvariable=self.model_status_var,
            bg=PANEL,
            fg=MUTED,
            font=("Malgun Gothic", 8),
        ).pack(side="right")

        results = tk.Frame(ai_panel, bg=PANEL)
        results.pack(fill="x")
        results.grid_columnconfigure(0, weight=2)
        results.grid_columnconfigure(1, weight=1)
        results.grid_columnconfigure(2, weight=1)
        primary = tk.Frame(results, bg="#251f38", padx=14, pady=10)
        primary.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(
            primary, text="최종 상태", bg="#251f38", fg=MUTED,
            font=("Malgun Gothic", 8)
        ).pack(anchor="w")
        self.primary_var = tk.StringVar(value="입력 대기")
        self.primary_label = tk.Label(
            primary,
            textvariable=self.primary_var,
            bg="#251f38",
            fg=MUTED,
            font=("Malgun Gothic", 16, "bold"),
        )
        self.primary_label.pack(anchor="w", pady=(4, 0))
        self.rf_result_var = tk.StringVar(value="꺼짐")
        self.cnn_result_var = tk.StringVar(value="꺼짐")
        for column, title, variable in (
            (1, "Random Forest", self.rf_result_var),
            (2, "1D CNN", self.cnn_result_var),
        ):
            box = tk.Frame(results, bg=PANEL_2, padx=12, pady=9)
            box.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 1 else 0)
            tk.Label(
                box, text=title, bg=PANEL_2, fg=MUTED,
                font=("Malgun Gothic", 9, "bold")
            ).pack(anchor="w")
            tk.Label(
                box, textvariable=variable, bg=PANEL_2, fg=TEXT,
                font=("Malgun Gothic", 12, "bold")
            ).pack(anchor="w", pady=(5, 0))

        metrics = tk.Frame(self.root, bg=BG)
        metrics.pack(fill="x", padx=24, pady=(0, 10))
        self.channel_vars: list[tk.StringVar] = []
        for index, color in enumerate(CHANNEL_COLORS):
            metrics.grid_columnconfigure(index, weight=1)
            box = tk.Frame(
                metrics, bg=PANEL, padx=13, pady=8,
                highlightthickness=1, highlightbackground="#30384b"
            )
            box.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 5, 5))
            tk.Label(
                box, text=f"A{index} RAW / 필터 RMS", bg=PANEL, fg=MUTED,
                font=("Malgun Gothic", 8, "bold")
            ).pack(anchor="w")
            variable = tk.StringVar(value="---")
            tk.Label(
                box, textvariable=variable, bg=PANEL, fg=color,
                font=("Segoe UI", 16, "bold")
            ).pack(anchor="w", pady=(2, 0))
            self.channel_vars.append(variable)
        metrics.grid_columnconfigure(3, weight=1)
        activity = tk.Frame(
            metrics, bg=PANEL, padx=13, pady=8,
            highlightthickness=1, highlightbackground="#30384b"
        )
        activity.grid(row=0, column=3, sticky="nsew", padx=(0, 0))
        tk.Label(
            activity, text="활성 점수 / 기준", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 8, "bold")
        ).pack(anchor="w")
        self.activation_var = tk.StringVar(value="---")
        tk.Label(
            activity, textvariable=self.activation_var, bg=PANEL, fg=WARNING,
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(3, 0))

        graph_panel = tk.Frame(
            self.root, bg=PANEL, highlightthickness=1, highlightbackground="#272e40"
        )
        graph_panel.pack(fill="both", expand=True, padx=24, pady=(0, 12))
        graph_header = tk.Frame(graph_panel, bg=PANEL)
        graph_header.pack(fill="x", padx=13, pady=8)
        tk.Label(
            graph_header, text="최근 2초 RAW ADC", bg=PANEL, fg=TEXT,
            font=("Malgun Gothic", 10, "bold")
        ).pack(side="left")
        tk.Label(
            graph_header,
            text="연결 직후 3초 동안 완전히 힘을 빼고 휴식 기준을 설정하세요",
            bg=PANEL,
            fg=WARNING,
            font=("Malgun Gothic", 8),
        ).pack(side="left", padx=12)
        self.graph = tk.Canvas(graph_panel, bg="#111620", highlightthickness=0)
        self.graph.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.graph.bind("<Configure>", lambda _event: self.draw_graph())

    def load_models(self) -> None:
        selected_mode = self.ai_mode_var.get()
        if self.loaded_mode is not None:
            if selected_mode != self.loaded_mode:
                messagebox.showinfo(
                    "재시작 필요",
                    "두 모델이 메모리에 함께 남지 않도록 모델 변경을 막았습니다.\n"
                    "프로그램을 닫았다가 다시 열고 다른 모델을 선택하세요.",
                )
            return

        output = model_output_directory()
        self.rf_model = None
        self.cnn_model = None
        self.torch_module = None
        metadata: dict[str, object]
        try:
            if selected_mode == "Random Forest":
                import joblib
                # The fitted estimator is restored dynamically by joblib.  Keep an
                # explicit branch-local import so PyInstaller bundles scikit-learn,
                # without importing it when the user runs only the CNN.
                from sklearn.ensemble import RandomForestClassifier  # noqa: F401

                package = joblib.load(output / "emg_raw_two_stage_rf.joblib")
                self.rf_model = package["model"]
                self.rf_model.n_jobs = 1
                metadata = dict(package["metadata"])
            elif selected_mode == "1D CNN":
                import torch
                from raw_cnn_model import RawEmg1DCNN

                torch.set_num_threads(1)
                checkpoint = torch.load(
                    output / "emg_raw_two_stage_cnn.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                metadata = dict(checkpoint["metadata"])
                labels = list(metadata["gesture_labels"])
                self.cnn_model = RawEmg1DCNN(len(labels))
                self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
                self.cnn_model.eval()
                self.torch_module = torch
            else:
                raise ValueError(f"지원하지 않는 모델: {selected_mode}")
        except Exception as exc:
            self.model_status_var.set(f"{selected_mode} 오류")
            messagebox.showerror("모델 로드 실패", f"{selected_mode}\n\n{exc}")
            return

        self.metadata = metadata
        self.gesture_labels = list(metadata["gesture_labels"])
        activation = dict(metadata["activation"])
        self.model_baseline_rms = np.asarray(
            activation["baseline_rms"], dtype=np.float32
        )
        self.activation_on = float(activation["on_ratio"])
        trained_off = float(activation["off_ratio"])
        # Enter rest a little more readily than the training-time gate.  The
        # upper bound preserves hysteresis, so activation does not chatter.
        self.activation_off = min(self.activation_on * 0.92, trained_off * 1.15)
        self.rf_confidence_threshold = float(
            dict(metadata["rf"])["confidence_threshold"]
        )
        rf_metadata = dict(metadata["rf"])
        self.rf_feature_version = str(rf_metadata.get("feature_version", "legacy_v1"))
        cnn = dict(metadata["cnn"])
        self.cnn_confidence_threshold = float(cnn["confidence_threshold"])
        self.cnn_scale = float(cnn["input_scale"])
        filter_metadata = dict(metadata.get("filter", {}))
        bandpass = list(filter_metadata.get("bandpass_hz", [FILTER_LOW_HZ, FILTER_HIGH_HZ]))
        self.filter_low_hz = float(bandpass[0])
        self.filter_high_hz = float(bandpass[1])
        self.notch_frequencies_hz = tuple(
            float(value) for value in filter_metadata.get("notch_hz", NOTCH_FREQUENCIES_HZ)
        )
        self.notch_half_width_hz = float(
            filter_metadata.get("notch_half_width_hz", NOTCH_HALF_WIDTH_HZ)
        )
        classification_metadata = dict(metadata.get("classification_preprocessing", {}))
        self.classification_normalization = str(
            classification_metadata.get("normalization", "none")
        )
        self.loaded_mode = selected_mode
        self.ai_mode_combo.configure(state="disabled")
        self.model_load_button.configure(text="선택 모델 실행 중", state="disabled")
        self.model_status_var.set(f"최종 5상태 {selected_mode}만 메모리에 로드됨")
        self.reset_prediction_state("데이터 대기")

    def refresh_ports(self) -> None:
        labels: list[str] = []
        self.port_map.clear()
        for port in list_ports.comports():
            identity = f"{port.description or ''} {port.hwid or ''}".lower()
            if "bluetooth" in identity or "bthenum" in identity:
                continue
            label = f"{port.device}  ·  {port.description or '장치 설명 없음'}"
            self.port_map[label] = port.device
            labels.append(label)
        self.port_combo["values"] = labels
        if self.port_var.get() not in labels:
            preferred = next(
                (
                    label for label in labels
                    if any(word in label.lower() for word in ("arduino", "ch340", "usb serial"))
                ),
                labels[0] if labels else "",
            )
            self.port_var.set(preferred)

    def toggle_connection(self) -> None:
        self.disconnect() if self.serial_port is not None else self.connect()

    def connect(self) -> None:
        if self.loaded_mode is None:
            messagebox.showwarning(
                "모델 선택",
                "Random Forest 또는 1D CNN을 선택하고 [선택 모델 켜기]를 먼저 누르세요.",
            )
            return
        device = self.port_map.get(self.port_var.get())
        if not device:
            messagebox.showwarning("포트 선택", "연결할 COM 포트를 선택하세요.")
            return
        try:
            port = serial.Serial(device, BAUD_RATE, timeout=0.1)
        except serial.SerialException as exc:
            messagebox.showerror(
                "연결 실패",
                "측정기·Serial Monitor 등 COM 포트를 쓰는 프로그램을 닫으세요."
                f"\n\n{exc}",
            )
            return
        self.serial_port = port
        self.reader_stop.clear()
        self.parser.reset()
        self.last_packet_at = 0.0
        self.last_sample_index = None
        self.total_samples = 0
        self.total_gaps = 0
        self.raw_buffer.clear()
        self.history.clear()
        self.reader_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.reader_thread.start()
        self.connect_button.configure(text="연결 해제", bg="#8f3344")
        self.connection_var.set(f"● {device} 연결됨")
        self.start_baseline_calibration(show_warning=False)

    def disconnect(self) -> None:
        self.reader_stop.set()
        port = self.serial_port
        self.serial_port = None
        if port is not None:
            try:
                port.close()
            except serial.SerialException:
                pass
        self.connect_button.configure(text="연결", bg="#6040a8")
        self.connection_var.set("● 연결 안 됨")
        self.calibrating_baseline = False
        self.reset_prediction_state("연결 대기")

    def read_serial(self) -> None:
        port = self.serial_port
        if port is None:
            return
        while not self.reader_stop.is_set():
            try:
                chunk = port.read(port.in_waiting or 1)
                if not chunk:
                    continue
                frames = self.parser.feed(chunk)
                if frames:
                    samples = [frame[:4] for frame in frames]
                    self.message_queue.put(("samples", samples))
            except (serial.SerialException, OSError) as exc:
                if not self.reader_stop.is_set():
                    self.message_queue.put(("error", str(exc)))
                return

    def process_messages(self) -> None:
        for _ in range(80):
            try:
                kind, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "samples" and isinstance(payload, list):
                self.handle_samples(payload)
            elif kind == "error":
                self.disconnect()
                messagebox.showerror("통신 오류", str(payload))
        self.root.after(30, self.process_messages)

    def handle_samples(self, samples: list[tuple[int, int, int, int]]) -> None:
        self.last_packet_at = time.monotonic()
        for sample_index, a0, a1, a2 in samples:
            if self.last_sample_index is not None:
                delta = (sample_index - self.last_sample_index) & 0xFFFFFFFF
                if delta > 1:
                    self.total_gaps += delta - 1
            self.last_sample_index = sample_index
            self.total_samples += 1
            self.raw_buffer.append((float(a0), float(a1), float(a2)))
            self.history.append((sample_index, a0, a1, a2))
            self.latest_values = (a0, a1, a2)
        self.samples_since_prediction += len(samples)
        if self.samples_since_prediction >= PREDICTION_HOP:
            self.samples_since_prediction %= PREDICTION_HOP
            self.run_inference()

    def start_baseline_calibration(self, show_warning: bool = True) -> None:
        if self.serial_port is None:
            if show_warning:
                messagebox.showwarning("연결 필요", "먼저 Arduino를 연결하세요.")
            return
        self.raw_buffer.clear()
        self.baseline_measurements.clear()
        self.live_baseline_rms = None
        self.calibrating_baseline = True
        self.active_state = False
        self.on_counter = 0
        self.off_counter = 0
        self.reset_prediction_state("휴식 기준 설정 중")
        self.set_status("3초 동안 손과 팔에 힘을 완전히 빼세요", WARNING)

    @staticmethod
    def smooth(current: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        if previous is None or len(previous) != len(current):
            return current
        return 0.60 * previous + 0.40 * current

    def reset_prediction_state(self, message: str) -> None:
        self.rf_probability_ema = None
        self.cnn_probability_ema = None
        self.stable_gesture_label = None
        self.switch_candidate = None
        self.switch_counter = 0
        self.latest_rf_prediction = message if self.loaded_mode == "Random Forest" else "꺼짐"
        self.latest_cnn_prediction = message if self.loaded_mode == "1D CNN" else "꺼짐"
        self.latest_rf_confidence = 0.0
        self.latest_cnn_confidence = 0.0
        if hasattr(self, "rf_result_var"):
            self.update_ai_display()

    def run_inference(self) -> None:
        if len(self.raw_buffer) < WINDOW_SAMPLES:
            return
        raw_window = np.asarray(list(self.raw_buffer)[-WINDOW_SAMPLES:], dtype=np.float32)
        filtered = preprocess_window(
            raw_window,
            low_hz=self.filter_low_hz,
            high_hz=self.filter_high_hz,
            notch_frequencies_hz=self.notch_frequencies_hz,
            notch_half_width_hz=self.notch_half_width_hz,
        )
        # A shorter RMS view lets release/rest be recognized without waiting
        # for the entire one-second classification window to clear.
        self.latest_rms = channel_rms(filtered[-ACTIVATION_GATE_SAMPLES:])
        classified = (
            normalize_for_classifier(filtered)
            if self.classification_normalization == "global_window_rms"
            else filtered
        )

        if self.calibrating_baseline:
            self.baseline_measurements.append(self.latest_rms.copy())
            if len(self.baseline_measurements) >= BASELINE_WINDOW_COUNT:
                self.live_baseline_rms = np.median(
                    np.stack(self.baseline_measurements), axis=0
                ).astype(np.float32)
                self.calibrating_baseline = False
                self.reset_prediction_state(REST_LABEL)
                self.set_status("휴식 기준 설정 완료 · 판정 준비", SUCCESS)
            return

        baseline = (
            self.live_baseline_rms
            if self.live_baseline_rms is not None
            else self.model_baseline_rms
        )
        self.activation_score = float(
            np.max(self.latest_rms / np.maximum(baseline, 0.05))
        )
        if not self.active_state:
            self.on_counter = self.on_counter + 1 if self.activation_score >= self.activation_on else 0
            if self.on_counter >= 2:
                self.active_state = True
                self.on_counter = 0
                self.off_counter = 0
                self.rf_probability_ema = None
                self.cnn_probability_ema = None
        else:
            self.off_counter = self.off_counter + 1 if self.activation_score <= self.activation_off else 0
            if self.off_counter >= ACTIVATION_OFF_CONFIRMATIONS:
                self.active_state = False
                self.off_counter = 0
                self.reset_prediction_state(REST_LABEL)

        if not self.active_state:
            if self.loaded_mode == "Random Forest":
                self.latest_rf_prediction = REST_LABEL
                self.latest_rf_confidence = 1.0
                self.latest_cnn_prediction = "꺼짐"
                self.latest_cnn_confidence = 0.0
            elif self.loaded_mode == "1D CNN":
                self.latest_cnn_prediction = REST_LABEL
                self.latest_cnn_confidence = 1.0
                self.latest_rf_prediction = "꺼짐"
                self.latest_rf_confidence = 0.0
            self.update_ai_display()
            return

        if self.loaded_mode == "Random Forest" and self.rf_model is not None:
            features = extract_rf_features(
                classified, self.rf_feature_version
            ).reshape(1, -1)
            probability = self.rf_model.predict_proba(features)[0]
            self.rf_probability_ema = self.smooth(probability, self.rf_probability_ema)
            label_probabilities = np.zeros(len(self.gesture_labels), dtype=np.float32)
            for position, class_index in enumerate(self.rf_model.classes_):
                label_probabilities[int(class_index)] = self.rf_probability_ema[position]
            (
                self.latest_rf_prediction,
                self.latest_rf_confidence,
            ) = self.stabilize_gesture_prediction(
                label_probabilities, self.rf_confidence_threshold
            )
        else:
            self.latest_rf_prediction = "꺼짐"
            self.latest_rf_confidence = 0.0

        if (
            self.loaded_mode == "1D CNN"
            and self.cnn_model is not None
            and self.torch_module is not None
        ):
            torch = self.torch_module
            cnn_input = np.clip(classified.T / max(self.cnn_scale, 1e-6), -6.0, 6.0)
            tensor = torch.from_numpy(cnn_input.reshape(1, 3, WINDOW_SAMPLES).astype(np.float32))
            with torch.no_grad():
                probability = torch.softmax(self.cnn_model(tensor), dim=1)[0].numpy()
            self.cnn_probability_ema = self.smooth(probability, self.cnn_probability_ema)
            (
                self.latest_cnn_prediction,
                self.latest_cnn_confidence,
            ) = self.stabilize_gesture_prediction(
                self.cnn_probability_ema, self.cnn_confidence_threshold
            )
        else:
            self.latest_cnn_prediction = "꺼짐"
            self.latest_cnn_confidence = 0.0
        self.update_ai_display()

    def stabilize_gesture_prediction(
        self, probabilities: np.ndarray, confidence_threshold: float
    ) -> tuple[str, float]:
        """Hold a gesture unless a clearly stronger challenger persists."""
        best = int(np.argmax(probabilities))
        raw_label = self.gesture_labels[best]
        raw_confidence = float(probabilities[best])
        if raw_confidence < confidence_threshold:
            raw_label = UNKNOWN_LABEL

        stable = self.stable_gesture_label
        if stable not in self.gesture_labels:
            if raw_label == UNKNOWN_LABEL:
                return UNKNOWN_LABEL, raw_confidence
            self.stable_gesture_label = raw_label
            self.switch_candidate = None
            self.switch_counter = 0
            return raw_label, raw_confidence

        stable_index = self.gesture_labels.index(stable)
        stable_confidence = float(probabilities[stable_index])
        if raw_label in (UNKNOWN_LABEL, stable):
            self.switch_candidate = None
            self.switch_counter = 0
            return stable, stable_confidence

        clearly_stronger = (
            raw_confidence >= confidence_threshold
            and raw_confidence >= stable_confidence + GESTURE_SWITCH_MARGIN
        )
        if clearly_stronger:
            if self.switch_candidate == raw_label:
                self.switch_counter += 1
            else:
                self.switch_candidate = raw_label
                self.switch_counter = 1
            if self.switch_counter >= GESTURE_SWITCH_CONFIRMATIONS:
                self.stable_gesture_label = raw_label
                self.switch_candidate = None
                self.switch_counter = 0
                return raw_label, raw_confidence
        else:
            self.switch_candidate = None
            self.switch_counter = 0
        return stable, stable_confidence

    @staticmethod
    def prediction_text(label: str, confidence: float) -> str:
        if label in (REST_LABEL, "대기", "데이터 대기", "연결 대기", "휴식 기준 설정 중"):
            return label
        if label == UNKNOWN_LABEL:
            return f"{label} · 최고 {confidence * 100:.0f}%"
        if label.endswith("움직임"):
            return f"{label} · {confidence * 100:.0f}%"
        return label

    def update_ai_display(self) -> None:
        self.rf_result_var.set(
            self.prediction_text(self.latest_rf_prediction, self.latest_rf_confidence)
        )
        self.cnn_result_var.set(
            self.prediction_text(self.latest_cnn_prediction, self.latest_cnn_confidence)
        )
        selected = UNKNOWN_LABEL
        if self.loaded_mode is None:
            selected = "모델 꺼짐"
            primary = "RF 또는 CNN을 선택해 켜세요"
        elif self.calibrating_baseline:
            selected = "휴식 기준 설정 중"
            primary = selected
        elif not self.active_state:
            selected = REST_LABEL
            primary = REST_LABEL
        elif self.loaded_mode == "Random Forest":
            selected = self.latest_rf_prediction
            primary = self.prediction_text(selected, self.latest_rf_confidence)
        else:
            selected = self.latest_cnn_prediction
            primary = self.prediction_text(selected, self.latest_cnn_confidence)
        self.primary_var.set(primary)
        color_map = {
            REST_LABEL: MUTED,
            "중지 움직임": CHANNEL_COLORS[1],
            "약지 움직임": CHANNEL_COLORS[2],
            "검지 움직임": CHANNEL_COLORS[0],
            "주먹 쥐기": "#ff9f43",
            UNKNOWN_LABEL: WARNING,
            "휴식 기준 설정 중": WARNING,
        }
        self.primary_label.configure(fg=color_map.get(selected, WARNING))

    def refresh_display(self) -> None:
        for index, variable in enumerate(self.channel_vars):
            variable.set(f"{self.latest_values[index]:4d}  /  {self.latest_rms[index]:6.2f}")
        state = "동작" if self.active_state else "휴식"
        self.activation_var.set(
            f"{self.activation_score:5.2f}  ·  켜짐 {self.activation_on:4.2f}"
            f" / 꺼짐 {self.activation_off:4.2f}  ·  {state}"
        )
        if self.serial_port is not None and self.last_packet_at > 0:
            if time.monotonic() - self.last_packet_at > 1.5:
                self.set_status("데이터가 들어오지 않습니다", DANGER)
            elif self.calibrating_baseline:
                remaining = max(0, BASELINE_WINDOW_COUNT - len(self.baseline_measurements))
                self.set_status(f"휴식 기준 설정 중 · 힘을 빼세요 · {remaining}/21", WARNING)
            else:
                self.set_status(
                    f"RAW 1kHz 수신 · 누락 {self.total_gaps:,} · 1초 창으로 0.1초마다 판정",
                    SUCCESS if self.total_gaps == 0 else WARNING,
                )
        self.draw_graph()
        self.root.after(100, self.refresh_display)

    def set_status(self, text: str, color: str) -> None:
        self.status_var.set(text)
        self.status_label.configure(fg=color)

    def draw_graph(self) -> None:
        canvas = self.graph
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 100 or height < 100:
            return
        canvas.delete("all")
        left, right, top, bottom = 50, width - 18, 16, height - 30
        plot_width = max(1, right - left)
        plot_height = max(1, bottom - top)
        for value in (0, 150, 300, 450, 600):
            y = bottom - value / 614.0 * plot_height
            canvas.create_line(left, y, right, y, fill=GRID)
            canvas.create_text(left - 8, y, text=str(value), fill=MUTED, anchor="e")
        history = list(self.history)
        if history:
            first_index = history[0][0]
            span = max(1, history[-1][0] - first_index)
            step = max(1, len(history) // 1000)
            for channel, color in enumerate(CHANNEL_COLORS):
                points: list[float] = []
                for sample_index, a0, a1, a2 in history[::step]:
                    value = (a0, a1, a2)[channel]
                    x = left + ((sample_index - first_index) / span) * plot_width
                    y = bottom - min(614, max(0, value)) / 614.0 * plot_height
                    points.extend((x, y))
                if len(points) >= 4:
                    canvas.create_line(*points, fill=color, width=1.5)
        else:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="RAW 데이터 대기 중",
                fill=MUTED,
                font=("Malgun Gothic", 12),
            )

    def close_app(self) -> None:
        self.disconnect()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RawEmgAiClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
