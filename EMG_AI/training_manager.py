"""Safe training-run staging, validation and model promotion."""

from __future__ import annotations

import json
import math
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from storage_integrity import scan_storage


EXPECTED_MODEL_FILES = (
    "emg_raw_two_stage_rf.joblib",
    "emg_raw_two_stage_cnn.pt",
    "model_metadata.json",
    "training_report.json",
    "training_report.txt",
    "dataset_preflight.json",
)


@contextmanager
def exclusive_training_lock(ai_dir: Path):
    """Allow only one trainer process to stage/promote models at a time."""
    lock_path = ai_dir / "training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "이미 다른 인공지능 학습이 실행 중입니다. 기존 학습이 끝난 뒤 다시 시도하세요."
        ) from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def project_directory_from(source_file: Path) -> Path:
    source_dir = source_file.resolve().parent
    for candidate in (source_dir.parent, source_dir, Path.cwd()):
        if (candidate / "raw_records").is_dir() and (candidate / "EMG_AI").is_dir():
            return candidate
    return source_dir.parent


def output_name_for_scope(scope: str) -> str:
    if scope == "latest":
        return "outputs_final_5class"
    if scope == "all_dates":
        return "outputs_all_raw_dates_5class"
    raise ValueError(f"지원하지 않는 학습 범위: {scope}")


def make_run_directory(ai_dir: Path, scope: str, now: datetime | None = None) -> Path:
    moment = now or datetime.now()
    day_dir = ai_dir / "training_runs" / moment.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    base_name = moment.strftime("%H%M%S") + f"_{scope}"
    for number in range(10_000):
        suffix = "" if number == 0 else f"_{number:02d}"
        candidate = day_dir / f"{base_name}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise OSError("새 학습 실행 폴더를 만들 수 없습니다.")


def validate_training_output(output_dir: Path) -> dict[str, Any]:
    missing = [name for name in EXPECTED_MODEL_FILES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("학습 산출물 누락: " + ", ".join(missing))
    for name in ("emg_raw_two_stage_rf.joblib", "emg_raw_two_stage_cnn.pt"):
        if (output_dir / name).stat().st_size < 10_000:
            raise RuntimeError(f"모델 파일 크기가 비정상적으로 작습니다: {name}")

    metadata = json.loads((output_dir / "model_metadata.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
    if int(metadata.get("sample_rate_hz", 0)) != 1_000:
        raise RuntimeError("모델 sample_rate_hz가 1000이 아닙니다.")
    if int(metadata.get("window_samples", 0)) <= 0:
        raise RuntimeError("모델 window_samples가 올바르지 않습니다.")
    if len(metadata.get("gesture_labels", [])) != 4 or metadata.get("rest_label") != "휴식":
        raise RuntimeError("모델의 5상태 레이블 구성이 올바르지 않습니다.")
    preflight = report.get("preflight", {})
    if not preflight.get("ready_to_train"):
        raise RuntimeError("데이터 사전검사를 통과하지 못한 모델입니다.")
    for model_name in ("rf", "cnn"):
        metrics = report.get(model_name, {}).get("end_to_end_four_state_test", {})
        for key in ("trial_accuracy", "trial_macro_f1"):
            value = float(metrics.get(key, math.nan))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{model_name} {key}가 올바르지 않습니다: {value}")
    return {"metadata": metadata, "report": report}


def validate_model_deserialization(output_dir: Path) -> None:
    import joblib
    import torch

    from raw_cnn_model import RawEmg1DCNN

    rf_package = joblib.load(output_dir / "emg_raw_two_stage_rf.joblib")
    rf_model = rf_package["model"]
    if int(getattr(rf_model, "n_features_in_", 0)) <= 0:
        raise RuntimeError("RF 모델 특징 개수가 올바르지 않습니다.")

    checkpoint = torch.load(
        output_dir / "emg_raw_two_stage_cnn.pt",
        map_location="cpu",
        weights_only=True,
    )
    metadata = checkpoint["metadata"]
    model = RawEmg1DCNN(len(metadata["gesture_labels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, int(metadata["window_samples"])))
    if tuple(output.shape) != (1, 4):
        raise RuntimeError(f"CNN 출력 모양이 올바르지 않습니다: {tuple(output.shape)}")


def _atomic_copy(source: Path, target: Path) -> None:
    incoming = target.with_name(target.name + ".incoming")
    if incoming.exists():
        incoming.unlink()
    shutil.copy2(source, incoming)
    os.replace(incoming, target)


def promote_training_run(
    run_dir: Path,
    ai_dir: Path,
    scope: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    validation = validate_training_output(run_dir)
    validate_model_deserialization(run_dir)
    moment = now or datetime.now()
    target = ai_dir / output_name_for_scope(scope)
    backup: Path | None = None
    if target.is_dir() and any(target.iterdir()):
        backup_root = ai_dir / "model_backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"{target.name}_{moment.strftime('%Y%m%d_%H%M%S')}"
        number = 1
        while backup.exists():
            backup = backup_root / (
                f"{target.name}_{moment.strftime('%Y%m%d_%H%M%S')}_{number:02d}"
            )
            number += 1
        shutil.copytree(target, backup)

    target.mkdir(parents=True, exist_ok=True)
    for source in run_dir.iterdir():
        if source.is_file() and not source.name.endswith((".tmp", ".incoming")):
            _atomic_copy(source, target / source.name)

    manifest = {
        "activated_at": moment.isoformat(timespec="seconds"),
        "scope": scope,
        "source_run": str(run_dir.resolve()),
        "active_directory": str(target.resolve()),
        "backup_directory": str(backup.resolve()) if backup else None,
        "created_at": validation["metadata"].get("created_at"),
        "trials": validation["metadata"].get("data", {}).get("trials", {}),
        "recommended_model": validation["metadata"].get("recommended_model"),
    }
    manifest_temp = target / "active_model_manifest.json.incoming"
    manifest_temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_temp, target / "active_model_manifest.json")
    return manifest


def _run_training_worker_unlocked(project_dir: Path, scope: str, run_dir: Path) -> dict[str, Any]:
    records_root = project_dir / "raw_records"
    audit = scan_storage(records_root, deep=True)
    audit_path = run_dir / "storage_audit_before_training.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["healthy"]:
        raise RuntimeError(
            f"저장소 검사에서 {audit['error_count']}개 오류가 발견되어 학습을 중단했습니다."
        )

    import sys
    import train_final_3finger

    arguments = [
        "train_final_3finger.py",
        "--data-dir",
        str(records_root),
        "--output-dir",
        str(run_dir),
    ]
    if scope == "all_dates":
        arguments.append("--all-dates")
    previous_arguments = sys.argv
    try:
        sys.argv = arguments
        train_final_3finger.main()
    finally:
        sys.argv = previous_arguments

    print("[검증] RF/CNN 파일을 실제로 다시 불러오는 중...", flush=True)
    print("[적용] 검증 후 기존 모델을 백업하고 새 모델을 활성화합니다.", flush=True)
    manifest = promote_training_run(run_dir, project_dir / "EMG_AI", scope)
    print("[완료] 새 RF/CNN 모델이 활성화되었습니다.", flush=True)
    return manifest


def run_training_worker(project_dir: Path, scope: str, run_dir: Path) -> dict[str, Any]:
    with exclusive_training_lock(project_dir / "EMG_AI"):
        return _run_training_worker_unlocked(project_dir, scope, run_dir)
