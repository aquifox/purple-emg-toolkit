"""One-click safe RF + 1D-CNN training UI for Purple Team EMG."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from storage_integrity import format_summary, scan_storage
from training_manager import (
    make_run_directory,
    output_name_for_scope,
    project_directory_from,
    run_training_worker,
)


BG = "#0e111a"
PANEL = "#181e2b"
PANEL_2 = "#222b3c"
TEXT = "#f3f5fb"
MUTED = "#9ca8bd"
PURPLE = "#815ac8"
SUCCESS = "#42d392"
WARNING = "#f0b84b"
DANGER = "#ff6377"
LABEL_ORDER = ("검지 움직임", "중지 움직임", "약지 움직임", "주먹 쥐기", "휴식")


def application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def project_directory() -> Path:
    return project_directory_from(application_directory() / "placeholder.py")


def worker_command(project_dir: Path, scope: str, run_dir: Path) -> list[str]:
    arguments = [
        "--worker",
        "--project-dir",
        str(project_dir),
        "--scope",
        scope,
        "--run-dir",
        str(run_dir),
    ]
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    return [sys.executable, str(Path(__file__).resolve()), *arguments]


class OneClickTrainerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("퍼플팀 EMG 원클릭 AI 학습기")
        self.root.configure(bg=BG)
        self.root.geometry("980x760")
        self.root.minsize(900, 680)
        self.project_dir = project_directory()
        self.records_root = self.project_dir / "raw_records"
        self.ai_dir = self.project_dir / "EMG_AI"
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_run_dir: Path | None = None
        self.current_scope: str | None = None

        self.scope_var = tk.StringVar(value="latest")
        self.status_var = tk.StringVar(value="데이터 확인 중")
        self.data_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="")
        self._configure_style()
        self._build_ui()
        self.refresh_data_summary()
        self.root.after(100, self.process_messages)
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Purple.Horizontal.TProgressbar",
            troughcolor=PANEL_2,
            background=PURPLE,
            bordercolor=PANEL_2,
            lightcolor=PURPLE,
            darkcolor=PURPLE,
        )

    def _button(self, parent: tk.Widget, text: str, command, color: str) -> tk.Button:
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
            padx=16,
            pady=10,
            font=("Malgun Gothic", 10, "bold"),
        )

    def _build_ui(self) -> None:
        title = tk.Frame(self.root, bg=BG)
        title.pack(fill="x", padx=24, pady=(20, 12))
        tk.Label(
            title,
            text="퍼플팀 EMG 원클릭 AI 학습기",
            bg=BG,
            fg=TEXT,
            font=("Malgun Gothic", 20, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title,
            text="데이터 검사 → RF·1D CNN 학습 → 검증 → 기존 모델 백업 → 새 모델 적용",
            bg=BG,
            fg=MUTED,
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", pady=(4, 0))

        data_panel = tk.Frame(self.root, bg=PANEL, highlightthickness=1, highlightbackground="#30384b")
        data_panel.pack(fill="x", padx=24, pady=8)
        tk.Label(
            data_panel,
            text="현재 RAW 데이터",
            bg=PANEL,
            fg=TEXT,
            font=("Malgun Gothic", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(13, 4))
        tk.Label(
            data_panel,
            textvariable=self.data_var,
            justify="left",
            bg=PANEL,
            fg="#d6dbea",
            font=("Malgun Gothic", 10),
        ).pack(anchor="w", padx=16, pady=(0, 10))
        self.refresh_button = self._button(data_panel, "데이터 다시 검사", self.refresh_data_summary, "#46516a")
        self.refresh_button.pack(anchor="e", padx=14, pady=(0, 12))

        scope_panel = tk.Frame(self.root, bg=PANEL, highlightthickness=1, highlightbackground="#30384b")
        scope_panel.pack(fill="x", padx=24, pady=8)
        tk.Label(
            scope_panel,
            text="학습 범위",
            bg=PANEL,
            fg=TEXT,
            font=("Malgun Gothic", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(13, 4))
        for value, title_text, detail in (
            ("latest", "가장 최근 날짜만 사용 (권장)", "전극 위치가 현재와 같은 데이터에 집중합니다."),
            ("all_dates", "모든 날짜 사용", "이전 세션까지 합쳐 날짜 변화에 대한 성능을 시험합니다."),
        ):
            row = tk.Frame(scope_panel, bg=PANEL)
            row.pack(fill="x", padx=14, pady=4)
            tk.Radiobutton(
                row,
                variable=self.scope_var,
                value=value,
                bg=PANEL,
                activebackground=PANEL,
                selectcolor=PANEL_2,
                fg=TEXT,
                activeforeground=TEXT,
            ).pack(side="left")
            text = tk.Frame(row, bg=PANEL)
            text.pack(side="left", fill="x")
            tk.Label(text, text=title_text, bg=PANEL, fg=TEXT, font=("Malgun Gothic", 10, "bold")).pack(anchor="w")
            tk.Label(text, text=detail, bg=PANEL, fg=MUTED, font=("Malgun Gothic", 9)).pack(anchor="w")

        action = tk.Frame(self.root, bg=BG)
        action.pack(fill="x", padx=24, pady=10)
        self.train_button = self._button(action, "RF + 1D CNN 새로 학습", self.start_training, PURPLE)
        self.train_button.pack(side="left")
        self.open_button = self._button(action, "학습 결과 폴더", self.open_active_output, "#46516a")
        self.open_button.pack(side="left", padx=10)
        tk.Label(action, textvariable=self.status_var, bg=BG, fg=WARNING, font=("Malgun Gothic", 10, "bold")).pack(side="right")

        self.progress = ttk.Progressbar(self.root, mode="indeterminate", style="Purple.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=24, pady=(0, 8))

        log_panel = tk.Frame(self.root, bg=PANEL, highlightthickness=1, highlightbackground="#30384b")
        log_panel.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        tk.Label(
            log_panel,
            text="진행 기록",
            bg=PANEL,
            fg=TEXT,
            font=("Malgun Gothic", 11, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 5))
        self.log = tk.Text(
            log_panel,
            bg="#10151f",
            fg="#d9dfed",
            insertbackground=TEXT,
            relief="flat",
            wrap="word",
            height=12,
            font=("Consolas", 9),
        )
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        tk.Label(
            log_panel,
            textvariable=self.result_var,
            bg=PANEL,
            fg=SUCCESS,
            justify="left",
            font=("Malgun Gothic", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(0, 10))

    def append_log(self, text: str) -> None:
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")

    def refresh_data_summary(self) -> None:
        report = scan_storage(self.records_root, deep=False)
        dates = " / ".join(f"{date}: {count}회" for date, count in report["date_counts"].items()) or "없음"
        labels = " / ".join(f"{label}: {report['label_counts'].get(label, 0)}회" for label in LABEL_ORDER)
        self.data_var.set(
            f"날짜별  {dates}\n동작별  {labels}\n"
            f"격리 {report['quarantined_trials']}회 · 임시파일 {report['partial_files']}개 · "
            f"저장 오류 {report['error_count']}개"
        )
        self.status_var.set("저장소 정상" if report["healthy"] else "저장 오류 확인 필요")

    def start_training(self) -> None:
        if self.process is not None:
            messagebox.showinfo("학습 진행 중", "현재 학습이 끝날 때까지 기다려 주세요.")
            return
        report = scan_storage(self.records_root, deep=False)
        if not report["healthy"]:
            messagebox.showerror("저장소 오류", format_summary(report))
            return
        scope = self.scope_var.get()
        self.current_scope = scope
        self.current_run_dir = make_run_directory(self.ai_dir, scope)
        command = worker_command(self.project_dir, scope, self.current_run_dir)
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.ai_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.process = None
            messagebox.showerror("학습 시작 실패", str(exc))
            return
        self.log.delete("1.0", "end")
        self.result_var.set("")
        self.append_log(f"학습 실행 폴더: {self.current_run_dir}")
        self.status_var.set("학습 진행 중")
        self.train_button.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.progress.start(12)
        self.reader_thread = threading.Thread(target=self.read_worker_output, daemon=True)
        self.reader_thread.start()

    def read_worker_output(self) -> None:
        process = self.process
        run_dir = self.current_run_dir
        if process is None:
            return
        log_file = run_dir / "training_console.log" if run_dir else None
        handle = log_file.open("a", encoding="utf-8") if log_file else None
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    if handle:
                        handle.write(line)
                        handle.flush()
                    self.messages.put(("log", line))
            code = process.wait()
            self.messages.put(("done", code))
        finally:
            if handle:
                handle.close()

    def process_messages(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.append_log(str(payload))
            elif kind == "done":
                self.finish_training(int(payload))
        self.root.after(100, self.process_messages)

    def finish_training(self, return_code: int) -> None:
        self.progress.stop()
        self.train_button.configure(state="normal")
        self.refresh_button.configure(state="normal")
        self.process = None
        if return_code != 0:
            self.status_var.set("학습 실패 · 기존 모델 유지")
            messagebox.showerror(
                "학습 실패",
                "새 모델을 적용하지 않았습니다. 진행 기록의 마지막 오류를 확인하세요.",
            )
            return
        scope = self.current_scope or self.scope_var.get()
        active = self.ai_dir / output_name_for_scope(scope)
        try:
            report = json.loads((active / "training_report.json").read_text(encoding="utf-8"))
            rf = report["rf"]["end_to_end_four_state_test"]
            cnn = report["cnn"]["end_to_end_four_state_test"]
            self.result_var.set(
                f"RF 정확도 {rf['trial_accuracy'] * 100:.1f}% · macro F1 {rf['trial_macro_f1']:.3f}\n"
                f"CNN 정확도 {cnn['trial_accuracy'] * 100:.1f}% · macro F1 {cnn['trial_macro_f1']:.3f}\n"
                "새 모델 적용 완료 · 판별기를 다시 열면 사용됩니다."
            )
        except Exception as exc:
            self.result_var.set(f"모델 적용 완료 · 결과 요약 읽기 실패: {exc}")
        self.status_var.set("학습 및 적용 완료")
        self.refresh_data_summary()
        messagebox.showinfo("완료", "RF와 1D CNN 학습·검증·백업·적용이 완료되었습니다.")

    def open_active_output(self) -> None:
        path = self.ai_dir / output_name_for_scope(self.scope_var.get())
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("폴더 열기 실패", str(exc))

    def close_app(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(
                "학습 진행 중",
                "창을 닫아도 학습 프로세스는 끝날 때까지 계속됩니다. 창을 닫을까요?",
            ):
                return
        self.root.destroy()


def worker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("latest", "all_dates"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print("[1/5] RAW 저장소 정밀검사를 시작합니다.", flush=True)
        run_training_worker(args.project_dir.resolve(), args.scope, args.run_dir.resolve())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[실패] {exc}", flush=True)
        raise SystemExit(1)


def main() -> None:
    if "--worker" in sys.argv:
        worker_main()
        return
    root = tk.Tk()
    OneClickTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
