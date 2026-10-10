"""Purple Team 3-channel 1 kHz RAW EMG acquisition program."""

from __future__ import annotations

import csv
import json
import os
import queue
import random
import re
import struct
import sys
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import serial
from serial.tools import list_ports


BAUD_RATE = 500_000
SAMPLE_RATE_HZ = 1_000
PACKET_SIZE = 14
SYNC = b"\xA5\x5A"
PAYLOAD_STRUCT = struct.Struct("<IHHH")
CRC_STRUCT = struct.Struct("<H")
GRAPH_SECONDS = 2.0
HISTORY_SAMPLES = 3_000
LABELS = ("검지 움직임", "중지 움직임", "약지 움직임", "휴식", "주먹 쥐기")

BG = "#0d1119"
PANEL = "#171d29"
PANEL_2 = "#202838"
GRID = "#30394d"
TEXT = "#f4f6fb"
MUTED = "#9ca8bd"
WARNING = "#f5b942"
SUCCESS = "#45d08b"
DANGER = "#ff6478"
CHANNEL_COLORS = ("#bd75ff", "#55a8ff", "#36d39b")


def crc16_ccitt_false(data: bytes | bytearray | memoryview) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_packet(sample_index: int, a0: int, a1: int, a2: int) -> bytes:
    payload = PAYLOAD_STRUCT.pack(sample_index & 0xFFFFFFFF, a0, a1, a2)
    return SYNC + payload + CRC_STRUCT.pack(crc16_ccitt_false(payload))


class RawPacketParser:
    """Streaming parser that can recover after noise, truncation or byte loss."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.crc_errors = 0
        self.discarded_bytes = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.crc_errors = 0
        self.discarded_bytes = 0

    def feed(self, chunk: bytes) -> list[tuple[int, int, int, int, bytes]]:
        self.buffer.extend(chunk)
        parsed: list[tuple[int, int, int, int, bytes]] = []
        while True:
            sync_position = self.buffer.find(SYNC)
            if sync_position < 0:
                keep = 1 if self.buffer.endswith(SYNC[:1]) else 0
                discard = len(self.buffer) - keep
                if discard > 0:
                    del self.buffer[:discard]
                    self.discarded_bytes += discard
                break
            if sync_position > 0:
                del self.buffer[:sync_position]
                self.discarded_bytes += sync_position
            if len(self.buffer) < PACKET_SIZE:
                break

            frame = bytes(self.buffer[:PACKET_SIZE])
            payload = frame[2:12]
            received_crc = CRC_STRUCT.unpack_from(frame, 12)[0]
            if crc16_ccitt_false(payload) != received_crc:
                del self.buffer[0]
                self.crc_errors += 1
                self.discarded_bytes += 1
                continue

            sample_index, a0, a1, a2 = PAYLOAD_STRUCT.unpack(payload)
            if a0 > 1023 or a1 > 1023 or a2 > 1023:
                del self.buffer[0]
                self.discarded_bytes += 1
                continue
            del self.buffer[:PACKET_SIZE]
            parsed.append((sample_index, a0, a1, a2, frame))
        return parsed


def application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def project_directory() -> Path:
    app_dir = application_directory()
    for candidate in (app_dir.parent, app_dir.parent.parent, app_dir, Path.cwd()):
        if (candidate / "EMG_측정_프로그램").is_dir() or (candidate / "records").is_dir():
            return candidate
    return app_dir.parent


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", value).strip(" .")
    return cleaned[:48] or "RAW_EMG"


def dated_records_directory(
    records_root: Path | None = None, when: datetime | None = None
) -> Path:
    root = records_root or (project_directory() / "raw_records")
    directory = root / (when or datetime.now()).strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def unique_record_base(directory: Path, desired_name: str) -> Path:
    """Return a base path that cannot overwrite a complete or partial trial."""
    suffixes = (".bin", ".csv", ".json", ".bin.part", ".csv.part", ".json.part")
    for duplicate_number in range(0, 10_000):
        tail = "" if duplicate_number == 0 else f"_중복{duplicate_number:02d}"
        candidate = directory / f"{desired_name}{tail}"
        if not any(candidate.with_suffix(suffix).exists() for suffix in suffixes):
            return candidate
    raise OSError("같은 이름의 기록이 너무 많아 새 파일명을 만들 수 없습니다.")


class RawEmgRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("퍼플팀 EMG RAW 측정기 · 1kHz × 3채널")
        self.root.configure(bg=BG)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        width = min(1280, max(1060, screen_width - 70))
        height = min(940, max(780, screen_height - 70))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1040, 740)

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.reader_stop = threading.Event()
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.parser = RawPacketParser()
        self.port_map: dict[str, str] = {}
        self.closing = False

        self.history: deque[tuple[int, int, int, int]] = deque(maxlen=HISTORY_SAMPLES)
        self.latest_values = (0, 0, 0)
        self.last_sample_index: int | None = None
        self.total_samples = 0
        self.total_gaps = 0
        self.out_of_order = 0
        self.connected_at = 0.0
        self.last_packet_at = 0.0
        self.rate_previous_samples = 0
        self.rate_previous_time = time.monotonic()
        self.measured_rate = 0.0
        self.receive_error_total_previous = 0
        self.last_receive_error_at = 0.0

        self.record_lock = threading.RLock()
        self.binary_file = None
        self.csv_file = None
        self.csv_writer = None
        self.record_base_path: Path | None = None
        self.binary_temp_path: Path | None = None
        self.csv_temp_path: Path | None = None
        self.json_temp_path: Path | None = None
        self.record_started_wall = ""
        self.record_started_at = 0.0
        self.record_start_index: int | None = None
        self.record_last_index: int | None = None
        self.record_rows = 0
        self.record_gaps = 0
        self.record_crc_start = 0
        self.record_discard_start = 0
        self.record_mode = ""
        self.current_phase = "IDLE"
        self.current_target = ""
        self.current_trial = 0
        self.record_events: list[dict[str, object]] = []

        self.protocol_running = False
        self.protocol_sequence: list[str] = []
        self.protocol_trial_position = 0
        self.protocol_phase_position = 0
        self.protocol_phase_started = 0.0
        self.protocol_after_id: str | None = None
        self.protocol_session_id = ""
        self.recording_session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.protocol_completed_trials = 0
        self.protocol_trial_completed = False
        self.protocol_phases = (
            ("BASELINE", 3.0),
            ("READY", 1.0),
            ("ACTION", 3.0),
            ("RECOVERY", 2.0),
        )
        self.hotkeys_down: set[str] = set()
        self.hotkey_release_jobs: dict[str, str] = {}
        self.participant_var = tk.StringVar(value="P01")
        self.arm_side_var = tk.StringVar(value="오른팔")
        self.attachment_var = tk.StringVar(
            value=datetime.now().strftime("ATT_%Y%m%d_01")
        )
        self.record_context: dict[str, str] = {}

        self._configure_styles()
        self._build_ui()
        self._bind_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.refresh_ports()
        self.root.after(35, self.process_messages)
        self.root.after(80, self.refresh_display)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.TCombobox", fieldbackground=PANEL_2, background=PANEL_2,
            foreground=TEXT, arrowcolor=TEXT, bordercolor="#3b4560"
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", PANEL_2)],
            foreground=[("readonly", TEXT)],
        )
        style.configure(
            "Dark.TSpinbox", fieldbackground=PANEL_2, background=PANEL_2,
            foreground=TEXT, arrowcolor=TEXT, bordercolor="#3b4560"
        )

    @staticmethod
    def _button(
        parent: tk.Widget, text: str, command, color: str, width: int = 10
    ) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, bg=color, fg="white",
            activebackground=color, activeforeground="white", relief="flat",
            bd=0, padx=10, pady=8, width=width,
            font=("Malgun Gothic", 9, "bold"), cursor="hand2"
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 12))
        tk.Label(
            header, text="퍼플팀 EMG RAW 측정기", bg=BG, fg=TEXT,
            font=("Malgun Gothic", 21, "bold")
        ).pack(side="left")
        tk.Label(
            header, text="1,000 Hz/채널 · A0/A1/A2 · AI/필터 없음",
            bg=BG, fg=WARNING, font=("Malgun Gothic", 10, "bold")
        ).pack(side="left", padx=16)
        self.connection_var = tk.StringVar(value="● 연결 안 됨")
        tk.Label(
            header, textvariable=self.connection_var, bg=BG, fg=MUTED,
            font=("Malgun Gothic", 10)
        ).pack(side="right")

        controls = tk.Frame(self.root, bg=PANEL, padx=14, pady=11)
        controls.pack(fill="x", padx=24, pady=(0, 10))
        tk.Label(
            controls, text="COM 포트", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 9, "bold")
        ).pack(side="left", padx=(2, 9))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            controls, textvariable=self.port_var, state="readonly",
            width=34, style="Dark.TCombobox"
        )
        self.port_combo.pack(side="left")
        self._button(controls, "새로고침", self.refresh_ports, "#30394c", 9).pack(
            side="left", padx=7
        )
        self.connect_button = self._button(
            controls, "연결", self.toggle_connection, "#6544ad", 9
        )
        self.connect_button.pack(side="left")
        self._button(
            controls, "저장 폴더", self.open_records_folder, "#34435a", 10
        ).pack(side="right")

        status_panel = tk.Frame(
            self.root, bg=PANEL_2, highlightthickness=1,
            highlightbackground="#31394d"
        )
        status_panel.pack(fill="x", padx=24, pady=(0, 10))
        self.status_var = tk.StringVar(
            value="새 RAW 바이너리 펌웨어를 업로드한 뒤 연결하세요."
        )
        self.status_label = tk.Label(
            status_panel, textvariable=self.status_var, bg=PANEL_2, fg=MUTED,
            font=("Malgun Gothic", 10, "bold"), anchor="w", pady=11
        )
        self.status_label.pack(fill="x", padx=16)

        metrics = tk.Frame(self.root, bg=BG)
        metrics.pack(fill="x", padx=24, pady=(0, 10))
        self.channel_vars: list[tk.StringVar] = []
        for index, color in enumerate(CHANNEL_COLORS):
            metrics.grid_columnconfigure(index, weight=1)
            box = tk.Frame(
                metrics, bg=PANEL, padx=14, pady=9,
                highlightthickness=1, highlightbackground="#30384b"
            )
            box.grid(
                row=0, column=index, sticky="nsew",
                padx=(0 if index == 0 else 5, 0 if index == 2 else 5)
            )
            tk.Label(
                box, text=f"A{index} RAW ADC", bg=PANEL, fg=MUTED,
                font=("Malgun Gothic", 9, "bold")
            ).pack(anchor="w")
            value_var = tk.StringVar(value="---")
            tk.Label(
                box, textvariable=value_var, bg=PANEL, fg=color,
                font=("Segoe UI", 20, "bold")
            ).pack(anchor="w", pady=(2, 0))
            self.channel_vars.append(value_var)

        metrics.grid_columnconfigure(3, weight=1)
        quality_box = tk.Frame(
            metrics, bg=PANEL, padx=14, pady=9,
            highlightthickness=1, highlightbackground="#30384b"
        )
        quality_box.grid(row=0, column=3, sticky="nsew", padx=(5, 0))
        tk.Label(
            quality_box, text="전송 품질", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 9, "bold")
        ).pack(anchor="w")
        self.quality_var = tk.StringVar(value="수신 대기")
        tk.Label(
            quality_box, textvariable=self.quality_var, bg=PANEL, fg=TEXT,
            justify="left", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(3, 0))

        record_panel = tk.Frame(
            self.root, bg=PANEL, padx=14, pady=10,
            highlightthickness=1, highlightbackground="#30384b"
        )
        record_panel.pack(fill="x", padx=24, pady=(0, 10))
        context_row = tk.Frame(record_panel, bg=PANEL)
        context_row.pack(fill="x", pady=(0, 9))
        tk.Label(
            context_row, text="참가자 ID", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 9, "bold")
        ).pack(side="left")
        tk.Entry(
            context_row, textvariable=self.participant_var, width=10,
            bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="flat",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=(6, 14), ipady=4)
        tk.Label(
            context_row, text="측정 팔", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 9, "bold")
        ).pack(side="left")
        ttk.Combobox(
            context_row, textvariable=self.arm_side_var,
            values=("오른팔", "왼팔"), state="readonly", width=8,
            style="Dark.TCombobox"
        ).pack(side="left", padx=(6, 14))
        tk.Label(
            context_row, text="전극 부착 ID", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 9, "bold")
        ).pack(side="left")
        tk.Entry(
            context_row, textvariable=self.attachment_var, width=18,
            bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="flat",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=(6, 12), ipady=4)
        tk.Label(
            context_row,
            text="전극을 다시 붙일 때 부착 ID 끝번호를 바꾸세요.",
            bg=PANEL, fg=WARNING, font=("Malgun Gothic", 9)
        ).pack(side="left")
        manual_row = tk.Frame(record_panel, bg=PANEL)
        manual_row.pack(fill="x")
        tk.Label(
            manual_row, text="수동 RAW 기록", bg=PANEL, fg=TEXT,
            font=("Malgun Gothic", 10, "bold")
        ).pack(side="left", padx=(0, 10))
        self.manual_buttons: list[tk.Button] = []
        for index, label in enumerate(LABELS):
            button = self._button(
                manual_row, f"{index + 1} · {label}",
                lambda chosen=label: self.toggle_manual_record(chosen),
                "#344765", 11
            )
            button.pack(side="left", padx=3)
            self.manual_buttons.append(button)
        self._button(
            manual_row, "자유 기록", lambda: self.toggle_manual_record("자유 기록"),
            "#28684d", 10
        ).pack(side="left", padx=3)
        self.stop_button = self._button(
            manual_row, "중지 · Space", self.stop_all_recording, "#81394a", 12
        )
        self.stop_button.pack(side="right")

        guide_row = tk.Frame(record_panel, bg=PANEL)
        guide_row.pack(fill="x", pady=(10, 0))
        tk.Label(
            guide_row, text="무작위 파일럿", bg=PANEL, fg=TEXT,
            font=("Malgun Gothic", 10, "bold")
        ).pack(side="left", padx=(0, 8))
        tk.Label(
            guide_row, text="동작당 반복", bg=PANEL, fg=MUTED,
            font=("Malgun Gothic", 9)
        ).pack(side="left")
        self.repetition_var = tk.IntVar(value=5)
        self.repetition_spin = ttk.Spinbox(
            guide_row, from_=1, to=30, textvariable=self.repetition_var,
            width=5, style="Dark.TSpinbox"
        )
        self.repetition_spin.pack(side="left", padx=(5, 8))
        self.guide_button = self._button(
            guide_row, "무작위 측정 시작", self.start_guided_protocol,
            "#6b4aa6", 15
        )
        self.guide_button.pack(side="left")
        tk.Label(
            guide_row,
            text="휴식 3초 → 준비 1초 → 동작 3초 → 회복 2초",
            bg=PANEL, fg=MUTED, font=("Malgun Gothic", 9)
        ).pack(side="left", padx=12)
        self.record_status_var = tk.StringVar(value="기록 안 함")
        tk.Label(
            guide_row, textvariable=self.record_status_var, bg=PANEL,
            fg=WARNING, font=("Malgun Gothic", 9, "bold")
        ).pack(side="right")

        prompt_panel = tk.Frame(
            self.root, bg="#241e36", padx=16, pady=9,
            highlightthickness=1, highlightbackground="#4c3b71"
        )
        prompt_panel.pack(fill="x", padx=24, pady=(0, 10))
        self.prompt_var = tk.StringVar(
            value="숫자키 1/2/3/4: 한 번 누르면 시작 · 같은 키를 다시 누르면 저장"
        )
        self.prompt_label = tk.Label(
            prompt_panel, textvariable=self.prompt_var, bg="#241e36",
            fg=MUTED, font=("Malgun Gothic", 15, "bold")
        )
        self.prompt_label.pack(side="left")
        self.countdown_var = tk.StringVar(value="")
        tk.Label(
            prompt_panel, textvariable=self.countdown_var, bg="#241e36",
            fg=WARNING, font=("Segoe UI", 17, "bold")
        ).pack(side="right")

        graph_panel = tk.Frame(
            self.root, bg=PANEL, highlightthickness=1,
            highlightbackground="#272e40"
        )
        graph_panel.pack(fill="both", expand=True, padx=24, pady=(0, 10))
        graph_header = tk.Frame(graph_panel, bg=PANEL)
        graph_header.pack(fill="x", padx=14, pady=8)
        tk.Label(
            graph_header, text="RAW ADC 파형", bg=PANEL, fg=TEXT,
            font=("Malgun Gothic", 10, "bold")
        ).pack(side="left")
        tk.Label(
            graph_header,
            text="최근 2초 · 세로축 0~1023 · 회색선은 1.5V 예상 중심(ADC 약 307)",
            bg=PANEL, fg=MUTED, font=("Malgun Gothic", 8)
        ).pack(side="left", padx=10)
        self._button(
            graph_header, "그래프 지우기", self.clear_graph, "#293246", 11
        ).pack(side="right")
        self.channel_visible_vars: list[tk.BooleanVar] = []
        for channel in range(2, -1, -1):
            visible_var = tk.BooleanVar(value=True)
            self.channel_visible_vars.insert(0, visible_var)
            tk.Checkbutton(
                graph_header,
                text=f"A{channel}",
                variable=visible_var,
                command=self.draw_graph,
                bg=PANEL,
                fg=CHANNEL_COLORS[channel],
                activebackground=PANEL,
                activeforeground=CHANNEL_COLORS[channel],
                selectcolor=PANEL_2,
                font=("Segoe UI", 9, "bold"),
                cursor="hand2",
                padx=4,
            ).pack(side="right", padx=(0, 3))
        self.graph = tk.Canvas(graph_panel, bg="#111620", highlightthickness=0)
        self.graph.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.graph.bind("<Configure>", lambda _event: self.draw_graph())

    def _bind_hotkeys(self) -> None:
        for index in range(len(LABELS)):
            key = str(index + 1)
            self.root.bind_all(
                f"<KeyPress-{index + 1}>",
                lambda event, chosen=index: self.handle_label_hotkey(event, chosen),
            )
            self.root.bind_all(
                f"<KeyRelease-{index + 1}>",
                lambda event, chosen=key: self.handle_hotkey_release(event, chosen),
            )
            self.root.bind_all(
                f"<KeyPress-KP_{index + 1}>",
                lambda event, chosen=index: self.handle_label_hotkey(event, chosen),
            )
            self.root.bind_all(
                f"<KeyRelease-KP_{index + 1}>",
                lambda event, chosen=key: self.handle_hotkey_release(event, chosen),
            )
        self.root.bind_all("<KeyPress-space>", self.handle_stop_hotkey)
        self.root.bind_all(
            "<KeyRelease-space>",
            lambda event: self.handle_hotkey_release(event, "space"),
        )

    def accept_hotkey_press(self, key: str) -> bool:
        pending_release = self.hotkey_release_jobs.pop(key, None)
        if pending_release is not None:
            try:
                self.root.after_cancel(pending_release)
            except tk.TclError:
                pass
            # Windows may generate a synthetic release/press pair while a key
            # is held. Cancel that release but do not treat the repeat as a new
            # physical press.
            return False
        if key in self.hotkeys_down:
            return False
        self.hotkeys_down.add(key)
        return True

    def handle_hotkey_release(self, _event: tk.Event, key: str) -> str:
        old_job = self.hotkey_release_jobs.pop(key, None)
        if old_job is not None:
            try:
                self.root.after_cancel(old_job)
            except tk.TclError:
                pass
        self.hotkey_release_jobs[key] = self.root.after(
            80, lambda chosen=key: self.finish_hotkey_release(chosen)
        )
        return "break"

    def finish_hotkey_release(self, key: str) -> None:
        self.hotkey_release_jobs.pop(key, None)
        self.hotkeys_down.discard(key)

    def handle_label_hotkey(self, event: tk.Event, index: int) -> str | None:
        if not self.accept_hotkey_press(str(index + 1)):
            return "break"
        self.toggle_manual_record(LABELS[index])
        return "break"

    def handle_stop_hotkey(self, event: tk.Event) -> str | None:
        if not self.accept_hotkey_press("space"):
            return "break"
        self.stop_all_recording()
        return "break"

    def refresh_ports(self) -> None:
        labels: list[str] = []
        self.port_map.clear()
        for port in list_ports.comports():
            description = port.description or "장치 설명 없음"
            identity = f"{description} {port.hwid or ''}".lower()
            if "bluetooth" in identity or "bthenum" in identity:
                continue
            label = f"{port.device}  ·  {description}"
            self.port_map[label] = port.device
            labels.append(label)
        self.port_combo["values"] = labels
        if self.port_var.get() not in labels:
            preferred = next(
                (
                    label for label in labels
                    if any(word in label.lower() for word in
                           ("arduino", "ch340", "usb serial", "usb-serial"))
                ),
                labels[0] if labels else "",
            )
            self.port_var.set(preferred)
        if not labels and self.serial_port is None:
            self.set_status("Arduino COM 포트를 찾지 못했습니다.", DANGER)

    def toggle_connection(self) -> None:
        self.disconnect() if self.serial_port is not None else self.connect()

    def connect(self) -> None:
        device = self.port_map.get(self.port_var.get())
        if not device:
            messagebox.showwarning("포트 선택", "연결할 COM 포트를 선택하세요.")
            return
        try:
            port = serial.Serial(
                device, BAUD_RATE, timeout=0.10, write_timeout=0.25
            )
            port.reset_input_buffer()
        except (serial.SerialException, OSError) as exc:
            messagebox.showerror(
                "연결 실패",
                "COM 포트를 열 수 없습니다. Arduino Serial Monitor와 다른 "
                f"측정 프로그램을 닫으세요.\n\n{exc}",
            )
            return

        self.serial_port = port
        self.reader_stop.clear()
        self.parser.reset()
        self.history.clear()
        self.latest_values = (0, 0, 0)
        self.last_sample_index = None
        self.total_samples = 0
        self.total_gaps = 0
        self.out_of_order = 0
        self.connected_at = time.monotonic()
        self.last_packet_at = 0.0
        self.rate_previous_samples = 0
        self.rate_previous_time = self.connected_at
        self.measured_rate = 0.0
        self.receive_error_total_previous = 0
        self.last_receive_error_at = 0.0
        self.recording_session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.reader_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.reader_thread.start()
        self.connect_button.configure(text="연결 해제", bg="#8e3445")
        self.connection_var.set(f"● {device} 연결됨")
        self.set_status(
            "Arduino가 재시작되는 중입니다. 약 1초 후 RAW 패킷을 확인합니다.",
            WARNING,
        )

    def disconnect(self) -> None:
        self.cancel_guided_protocol(stop_record=False)
        if self.is_recording:
            self.stop_recording()
        self.reader_stop.set()
        port = self.serial_port
        self.serial_port = None
        if port is not None:
            try:
                port.close()
            except (serial.SerialException, OSError):
                pass
        self.connect_button.configure(text="연결", bg="#6544ad")
        self.connection_var.set("● 연결 안 됨")
        self.set_status("Arduino 연결이 해제되었습니다.", MUTED)

    @property
    def is_recording(self) -> bool:
        return self.csv_file is not None

    def read_serial(self) -> None:
        port = self.serial_port
        if port is None:
            return
        try:
            while not self.reader_stop.is_set():
                waiting = port.in_waiting
                chunk = port.read(min(max(waiting, 1), 8192))
                if not chunk:
                    continue
                samples = self.parser.feed(chunk)
                if samples:
                    self.consume_samples(samples)
        except (serial.SerialException, OSError) as exc:
            if not self.reader_stop.is_set():
                self.message_queue.put(("error", str(exc)))

    def consume_samples(
        self, samples: list[tuple[int, int, int, int, bytes]]
    ) -> None:
        display_batch: list[tuple[int, int, int, int]] = []
        for sample_index, a0, a1, a2, frame in samples:
            if self.last_sample_index is not None:
                delta = (sample_index - self.last_sample_index) & 0xFFFFFFFF
                if 1 < delta < 0x80000000:
                    self.total_gaps += delta - 1
                elif delta == 0 or delta >= 0x80000000:
                    self.out_of_order += 1
            self.last_sample_index = sample_index
            self.total_samples += 1
            display_batch.append((sample_index, a0, a1, a2))
            self.write_record_sample(sample_index, a0, a1, a2, frame)
        self.last_packet_at = time.monotonic()
        self.message_queue.put(("samples", display_batch))

    def write_record_sample(
        self, sample_index: int, a0: int, a1: int, a2: int, frame: bytes
    ) -> None:
        with self.record_lock:
            if self.csv_file is None or self.binary_file is None or self.csv_writer is None:
                return
            if self.record_start_index is None:
                self.record_start_index = sample_index
            if self.record_last_index is not None:
                delta = (sample_index - self.record_last_index) & 0xFFFFFFFF
                if 1 < delta < 0x80000000:
                    self.record_gaps += delta - 1
            self.record_last_index = sample_index
            elapsed = (
                (sample_index - self.record_start_index) & 0xFFFFFFFF
            ) / SAMPLE_RATE_HZ
            self.binary_file.write(frame)
            self.csv_writer.writerow(
                [
                    sample_index, f"{elapsed:.3f}", a0, a1, a2,
                    self.current_phase, self.current_target, self.current_trial,
                ]
            )
            self.record_rows += 1
            if self.record_rows % SAMPLE_RATE_HZ == 0:
                self.binary_file.flush()
                self.csv_file.flush()

    def process_messages(self) -> None:
        if self.closing:
            return
        for _ in range(100):
            try:
                kind, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "samples":
                batch = payload
                if isinstance(batch, list):
                    self.history.extend(batch)
                    if batch:
                        _, a0, a1, a2 = batch[-1]
                        self.latest_values = (a0, a1, a2)
            elif kind == "error":
                error = str(payload)
                self.disconnect()
                messagebox.showerror("수신 오류", f"Arduino 연결이 끊겼습니다.\n\n{error}")
        self.root.after(35, self.process_messages)

    def start_recording(
        self,
        label: str,
        mode: str,
        *,
        session_id: str = "",
        trial_id: int = 0,
    ) -> bool:
        if self.serial_port is None or self.last_packet_at <= 0:
            messagebox.showwarning("수신 필요", "먼저 Arduino를 연결하고 RAW 수신을 확인하세요.")
            return False
        if self.is_recording:
            self.stop_recording()

        now = datetime.now()
        directory = dated_records_directory(when=now)
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if mode == "guided_trial":
            desired_name = (
                f"RAW_EMG_{safe_name(label)}_무작위_{safe_name(session_id)}_"
                f"회차{trial_id:03d}"
            )
        else:
            desired_name = f"RAW_EMG_{safe_name(label)}_{timestamp}"
        try:
            base = unique_record_base(directory, desired_name)
        except OSError as exc:
            messagebox.showerror("저장 실패", str(exc))
            return False
        binary_temp_path = base.with_suffix(".bin.part")
        csv_temp_path = base.with_suffix(".csv.part")
        json_temp_path = base.with_suffix(".json.part")
        binary_file = None
        csv_file = None
        try:
            binary_file = binary_temp_path.open("xb", buffering=64 * 1024)
            csv_file = csv_temp_path.open(
                "x", newline="", encoding="utf-8-sig", buffering=1024 * 1024
            )
        except OSError as exc:
            for handle in (binary_file, csv_file):
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
            for partial in (binary_temp_path, csv_temp_path, json_temp_path):
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
            messagebox.showerror("저장 실패", f"기록 파일을 만들 수 없습니다.\n\n{exc}")
            return False

        with self.record_lock:
            self.binary_file = binary_file
            self.csv_file = csv_file
            self.csv_writer = csv.writer(csv_file)
            self.csv_writer.writerow(
                [
                    "sample_index", "elapsed_seconds", "a0_adc", "a1_adc",
                    "a2_adc", "phase", "target_label", "trial_id",
                ]
            )
            self.record_base_path = base
            self.binary_temp_path = binary_temp_path
            self.csv_temp_path = csv_temp_path
            self.json_temp_path = json_temp_path
            self.record_started_wall = datetime.now().isoformat(timespec="milliseconds")
            self.record_started_at = time.monotonic()
            self.record_start_index = None
            self.record_last_index = None
            self.record_rows = 0
            self.record_gaps = 0
            self.record_crc_start = self.parser.crc_errors
            self.record_discard_start = self.parser.discarded_bytes
            self.record_mode = mode
            self.record_context = {
                "participant_id": self.participant_var.get().strip() or "P01",
                "arm_side": self.arm_side_var.get().strip() or "미기록",
                "electrode_attachment_id": (
                    self.attachment_var.get().strip() or "미기록"
                ),
            }
            self.current_phase = "MANUAL" if mode == "manual" else "BASELINE"
            self.current_target = label
            self.current_trial = trial_id
            self.record_events = []
            self.add_record_event("RECORD_START", label)

        self.record_status_var.set(f"● 기록 중 · {label}")
        return True

    def stop_recording(self) -> Path | None:
        save_error: OSError | None = None
        with self.record_lock:
            if self.csv_file is None:
                return None
            self.add_record_event("RECORD_STOP", self.current_target)
            base = self.record_base_path
            metadata = {
                "format_version": 3,
                "created_at": self.record_started_wall,
                "stopped_at": datetime.now().isoformat(timespec="milliseconds"),
                "mode": self.record_mode,
                **self.record_context,
                "recording_session_id": self.recording_session_id,
                "protocol_session_id": (
                    self.protocol_session_id if self.record_mode == "guided_trial" else None
                ),
                "protocol_trial_number": self.current_trial or None,
                "protocol_total_trials": len(self.protocol_sequence) or None,
                "protocol_trial_completed": (
                    self.protocol_trial_completed if self.record_mode == "guided_trial" else None
                ),
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "baud_rate": BAUD_RATE,
                "channels": ["A0", "A1", "A2"],
                "adc_bits": 10,
                "adc_reference": "Arduino DEFAULT (nominal 5 V)",
                "packet": {
                    "size_bytes": PACKET_SIZE,
                    "sync_hex": SYNC.hex().upper(),
                    "endianness": "little",
                    "payload": "uint32 index, uint16 A0, uint16 A1, uint16 A2",
                    "crc": "CRC-16/CCITT-FALSE over payload",
                    "binary_contains": "validated complete packets only",
                },
                "rows": self.record_rows,
                "first_sample_index": self.record_start_index,
                "last_sample_index": self.record_last_index,
                "missing_sample_groups": self.record_gaps,
                "crc_errors_during_record": self.parser.crc_errors - self.record_crc_start,
                "discarded_bytes_during_record": (
                    self.parser.discarded_bytes - self.record_discard_start
                ),
                "events": list(self.record_events),
                "files": {
                    "binary": f"{base.name}.bin" if base else None,
                    "csv": f"{base.name}.csv" if base else None,
                },
            }
            binary_file = self.binary_file
            csv_file = self.csv_file
            binary_temp_path = self.binary_temp_path
            csv_temp_path = self.csv_temp_path
            json_temp_path = self.json_temp_path
            self.binary_file = None
            self.csv_file = None
            self.csv_writer = None
            try:
                binary_file.flush()
                os.fsync(binary_file.fileno())
                binary_file.close()
                csv_file.flush()
                os.fsync(csv_file.fileno())
                csv_file.close()
                if base is None or binary_temp_path is None or csv_temp_path is None or json_temp_path is None:
                    raise OSError("저장 경로 상태가 올바르지 않습니다.")
                with json_temp_path.open("x", encoding="utf-8") as metadata_file:
                    json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
                    metadata_file.flush()
                    os.fsync(metadata_file.fileno())
                os.replace(binary_temp_path, base.with_suffix(".bin"))
                os.replace(csv_temp_path, base.with_suffix(".csv"))
                os.replace(json_temp_path, base.with_suffix(".json"))
            except OSError as exc:
                save_error = exc
                for handle in (binary_file, csv_file):
                    try:
                        if not handle.closed:
                            handle.close()
                    except OSError:
                        pass
            if save_error is not None and base is not None:
                failed_directory = base.parent / "저장실패"
                failed_directory.mkdir(parents=True, exist_ok=True)
                for suffix in (".bin", ".csv", ".json", ".bin.part", ".csv.part", ".json.part"):
                    source = base.with_suffix(suffix)
                    if source.exists():
                        target = failed_directory / source.name
                        if target.exists():
                            target = failed_directory / f"{base.name}_{int(time.time())}{suffix}"
                        try:
                            source.replace(target)
                        except OSError:
                            pass
            elif base is not None:
                try:
                    if self.record_mode == "guided_trial" and not self.protocol_trial_completed:
                        incomplete_directory = base.parent / "미완료"
                        incomplete_directory.mkdir(parents=True, exist_ok=True)
                        moved_base = unique_record_base(incomplete_directory, base.name)
                        for suffix in (".bin", ".csv", ".json"):
                            base.with_suffix(suffix).replace(moved_base.with_suffix(suffix))
                        base = moved_base
                except OSError as exc:
                    save_error = exc

            rows = self.record_rows
            gaps = self.record_gaps
            self.record_base_path = None
            self.binary_temp_path = None
            self.csv_temp_path = None
            self.json_temp_path = None
            self.record_mode = ""
            self.current_phase = "IDLE"
            self.current_target = ""
            self.current_trial = 0
        if save_error is not None:
            self.record_status_var.set("저장 실패 · 저장실패 폴더를 확인하세요")
            if not self.closing:
                messagebox.showerror(
                    "저장 실패",
                    "기록을 완성 파일로 확정하지 못했습니다.\n"
                    "남은 파일은 저장실패 폴더로 격리했습니다.\n\n"
                    f"{save_error}",
                )
            return None
        self.record_status_var.set(f"저장 완료 · {rows:,}샘플 · 누락 {gaps:,}")
        return base

    def add_record_event(self, phase: str, target: str) -> None:
        sample_index = self.last_sample_index
        elapsed = None
        if sample_index is not None and self.record_start_index is not None:
            elapsed = ((sample_index - self.record_start_index) & 0xFFFFFFFF) / SAMPLE_RATE_HZ
        self.record_events.append(
            {
                "sample_index": sample_index,
                "elapsed_seconds": elapsed,
                "phase": phase,
                "target_label": target,
                "trial_id": self.current_trial,
                "computer_time": datetime.now().isoformat(timespec="milliseconds"),
            }
        )

    def toggle_manual_record(self, label: str) -> None:
        if self.protocol_running:
            messagebox.showwarning("파일럿 진행 중", "먼저 무작위 파일럿을 중지하세요.")
            return
        if self.is_recording:
            if self.current_target == label and self.record_mode == "manual":
                self.stop_recording()
                self.prompt_var.set("수동 기록 저장 완료")
                self.countdown_var.set("")
                return
            self.stop_recording()
        if self.start_recording(label, "manual"):
            self.prompt_var.set(f"수동 기록 중 · {label}")
            self.prompt_label.configure(fg=WARNING)

    def stop_all_recording(self) -> None:
        if self.protocol_running:
            self.cancel_guided_protocol(stop_record=True)
            return
        if self.is_recording:
            self.stop_recording()
            self.prompt_var.set("기록을 저장하고 중지했습니다")
            self.prompt_label.configure(fg=MUTED)
            self.countdown_var.set("")

    def build_balanced_sequence(self, repetitions: int) -> list[str]:
        generator = random.SystemRandom()
        sequence: list[str] = []
        for _ in range(repetitions):
            block = list(LABELS)
            generator.shuffle(block)
            while sequence and block[0] == sequence[-1]:
                generator.shuffle(block)
            sequence.extend(block)
        return sequence

    def start_guided_protocol(self) -> None:
        if self.protocol_running:
            self.cancel_guided_protocol(stop_record=True)
            return
        try:
            repetitions = max(1, min(30, int(self.repetition_var.get())))
        except (TypeError, ValueError):
            repetitions = 5
            self.repetition_var.set(repetitions)
        sequence = self.build_balanced_sequence(repetitions)
        if self.serial_port is None or self.last_packet_at <= 0:
            messagebox.showwarning("수신 필요", "먼저 Arduino를 연결하고 RAW 수신을 확인하세요.")
            return
        self.protocol_sequence = sequence
        self.protocol_trial_position = 0
        self.protocol_phase_position = 0
        self.protocol_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.protocol_completed_trials = 0
        self.protocol_trial_completed = False
        self.protocol_running = True
        self.guide_button.configure(text="파일럿 중지", bg="#8d3447")
        self.begin_protocol_phase()

    def begin_protocol_phase(self) -> None:
        if not self.protocol_running:
            return
        if self.protocol_trial_position >= len(self.protocol_sequence):
            self.complete_guided_protocol()
            return
        phase, _duration = self.protocol_phases[self.protocol_phase_position]
        target = self.protocol_sequence[self.protocol_trial_position]
        self.current_trial = self.protocol_trial_position + 1
        if phase == "BASELINE" and not self.is_recording:
            self.protocol_trial_completed = False
            if not self.start_recording(
                target,
                "guided_trial",
                session_id=self.protocol_session_id,
                trial_id=self.current_trial,
            ):
                self.cancel_guided_protocol(stop_record=False)
                return
        self.current_phase = phase
        self.current_target = target
        self.protocol_phase_started = time.monotonic()
        with self.record_lock:
            self.add_record_event(phase, target)
        self.update_protocol_prompt(phase, target)
        self.schedule_protocol_tick()

    def schedule_protocol_tick(self) -> None:
        if self.protocol_after_id is not None:
            self.root.after_cancel(self.protocol_after_id)
        self.protocol_after_id = self.root.after(50, self.protocol_tick)

    def protocol_tick(self) -> None:
        self.protocol_after_id = None
        if not self.protocol_running:
            return
        phase, duration = self.protocol_phases[self.protocol_phase_position]
        elapsed = time.monotonic() - self.protocol_phase_started
        remaining = max(0.0, duration - elapsed)
        self.countdown_var.set(f"{remaining:0.1f}초")
        if elapsed >= duration:
            if phase == "RECOVERY":
                self.protocol_trial_completed = True
                self.protocol_completed_trials += 1
                self.stop_recording()
                self.protocol_phase_position = 0
                self.protocol_trial_position += 1
            else:
                self.protocol_phase_position += 1
            self.begin_protocol_phase()
        else:
            self.schedule_protocol_tick()

    def update_protocol_prompt(self, phase: str, target: str) -> None:
        trial = self.protocol_trial_position + 1
        total = len(self.protocol_sequence)
        if phase == "BASELINE":
            text, color = f"[{trial}/{total}] 힘을 완전히 빼세요", MUTED
        elif phase == "READY":
            text, color = f"[{trial}/{total}] 준비 · 다음 동작: {target}", WARNING
        elif phase == "ACTION":
            text = f"[{trial}/{total}] 지금: {target}"
            color = SUCCESS if target != "휴식" else "#b9c1d1"
        else:
            text, color = f"[{trial}/{total}] 동작 종료 · 다시 힘 빼기", MUTED
        self.prompt_var.set(text)
        self.prompt_label.configure(fg=color)
        self.record_status_var.set(f"● 파일럿 기록 중 · {trial}/{total}")

    def complete_guided_protocol(self) -> None:
        self.protocol_running = False
        self.protocol_after_id = None
        self.guide_button.configure(text="무작위 측정 시작", bg="#6b4aa6")
        self.countdown_var.set("")
        self.prompt_var.set(
            f"무작위 파일럿 완료 · {self.protocol_completed_trials}개 회차 파일 저장됨"
        )
        self.prompt_label.configure(fg=SUCCESS)
        if self.is_recording:
            self.stop_recording()

    def cancel_guided_protocol(self, stop_record: bool) -> None:
        if self.protocol_after_id is not None:
            try:
                self.root.after_cancel(self.protocol_after_id)
            except tk.TclError:
                pass
            self.protocol_after_id = None
        was_running = self.protocol_running
        self.protocol_running = False
        if hasattr(self, "guide_button"):
            self.guide_button.configure(text="무작위 측정 시작", bg="#6b4aa6")
        if stop_record and self.is_recording:
            self.stop_recording()
        if was_running and hasattr(self, "prompt_var"):
            self.prompt_var.set("무작위 파일럿을 중지하고 현재 파일을 저장했습니다")
            self.prompt_label.configure(fg=WARNING)
            self.countdown_var.set("")

    def refresh_display(self) -> None:
        if self.closing:
            return
        now = time.monotonic()
        elapsed = now - self.rate_previous_time
        if elapsed >= 0.8:
            delta = self.total_samples - self.rate_previous_samples
            instant_rate = delta / elapsed
            self.measured_rate = (
                instant_rate if self.measured_rate <= 0
                else 0.65 * self.measured_rate + 0.35 * instant_rate
            )
            self.rate_previous_samples = self.total_samples
            self.rate_previous_time = now

        window = list(self.history)[-250:]
        for channel, value_var in enumerate(self.channel_vars):
            if window:
                values = [sample[channel + 1] for sample in window]
                mean = sum(values) / len(values)
                peak_to_peak = max(values) - min(values)
                value_var.set(
                    f"{self.latest_values[channel]:4d}   평균 {mean:5.1f}   P2P {peak_to_peak:3d}"
                )
            else:
                value_var.set("---")

        self.quality_var.set(
            f"{self.measured_rate:7.1f} sample/s\n"
            f"누락 {self.total_gaps:,} · CRC {self.parser.crc_errors:,}"
        )
        receive_error_total = self.total_gaps + self.parser.crc_errors
        if receive_error_total > self.receive_error_total_previous:
            self.last_receive_error_at = now
        self.receive_error_total_previous = receive_error_total
        if self.serial_port is not None:
            if self.last_packet_at <= 0 and now - self.connected_at > 3.0:
                self.set_status(
                    "RAW 패킷이 없습니다. 새 바이너리 펌웨어와 500000 baud를 확인하세요.",
                    DANGER,
                )
            elif self.last_packet_at > 0 and now - self.last_packet_at > 1.0:
                self.set_status("RAW 데이터 수신이 1초 이상 멈췄습니다.", DANGER)
            elif self.total_samples > 0:
                if self.last_receive_error_at > 0 and now - self.last_receive_error_at < 3.0:
                    self.set_status(
                        f"최근 수신 오류 감지 · 누락 {self.total_gaps:,} · "
                        f"CRC {self.parser.crc_errors:,} (증가 여부 확인)",
                        WARNING,
                    )
                elif 970 <= self.measured_rate <= 1030:
                    if receive_error_total:
                        self.set_status(
                            f"RAW 수신 정상 · 이전 누적 오류 {receive_error_total}건은 더 이상 증가하지 않음",
                            SUCCESS,
                        )
                    else:
                        self.set_status(
                            "RAW 수신 정상 · 1kHz/채널 · CRC 및 sample_index 정상",
                            SUCCESS,
                        )
                else:
                    self.set_status(
                        f"RAW 수신 중 · 현재 측정 속도 {self.measured_rate:.1f} sample/s",
                        WARNING,
                    )
        self.draw_graph()
        self.root.after(80, self.refresh_display)

    def set_status(self, text: str, color: str) -> None:
        self.status_var.set(text)
        self.status_label.configure(fg=color)

    def clear_graph(self) -> None:
        self.history.clear()
        self.draw_graph()

    def draw_graph(self) -> None:
        canvas = self.graph
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 120 or height < 120:
            return
        canvas.delete("all")
        left, right, top, bottom = 50, width - 16, 14, height - 28
        plot_width = max(1, right - left)
        plot_height = max(1, bottom - top)
        for value in (0, 256, 512, 768, 1023):
            y = bottom - value / 1023.0 * plot_height
            canvas.create_line(left, y, right, y, fill=GRID)
            canvas.create_text(
                left - 8, y, text=str(value), fill=MUTED,
                font=("Segoe UI", 8), anchor="e"
            )
        center_y = bottom - 307 / 1023.0 * plot_height
        canvas.create_line(left, center_y, right, center_y, fill="#677188", dash=(4, 4))

        history = list(self.history)
        if history:
            newest_index = history[-1][0]
            oldest_allowed = newest_index - int(GRAPH_SECONDS * SAMPLE_RATE_HZ)
            visible = [sample for sample in history if sample[0] >= oldest_allowed]
            stride = max(1, len(visible) // max(1, int(plot_width)))
            visible = visible[::stride]
            for channel, color in enumerate(CHANNEL_COLORS):
                if not self.channel_visible_vars[channel].get():
                    continue
                points: list[float] = []
                for sample_index, a0, a1, a2 in visible:
                    age_samples = newest_index - sample_index
                    x = right - age_samples / (GRAPH_SECONDS * SAMPLE_RATE_HZ) * plot_width
                    value = (a0, a1, a2)[channel]
                    y = bottom - value / 1023.0 * plot_height
                    points.extend((x, y))
                if len(points) >= 4:
                    canvas.create_line(*points, fill=color, width=1)
        else:
            canvas.create_text(
                (left + right) / 2, (top + bottom) / 2,
                text="RAW 데이터 수신 대기", fill=MUTED,
                font=("Malgun Gothic", 13)
            )

        for seconds_ago in (0.0, 0.5, 1.0, 1.5, 2.0):
            x = right - seconds_ago / GRAPH_SECONDS * plot_width
            canvas.create_text(
                x, bottom + 15,
                text="지금" if seconds_ago == 0 else f"-{seconds_ago:g}s",
                fill=MUTED, font=("Malgun Gothic", 8)
            )

    def open_records_folder(self) -> None:
        directory = project_directory() / "raw_records"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(directory)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("폴더 열기 실패", str(exc))

    def close_app(self) -> None:
        self.closing = True
        for job in self.hotkey_release_jobs.values():
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        self.hotkey_release_jobs.clear()
        self.hotkeys_down.clear()
        self.cancel_guided_protocol(stop_record=True)
        if self.is_recording:
            self.stop_recording()
        self.reader_stop.set()
        port = self.serial_port
        self.serial_port = None
        if port is not None:
            try:
                port.close()
            except (serial.SerialException, OSError):
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RawEmgRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
