"""Load and quality-check dated 1 kHz, 3-channel RAW EMG trials.

Manual trials and completed randomized-protocol trials are accepted.  Partial,
failed and explicitly quarantined recordings are excluded before training.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raw_two_stage_common import (
    FINAL_FILTER_HIGH_HZ,
    FINAL_NOTCH_FREQUENCIES_HZ,
    FILTER_LOW_HZ,
    GESTURE_LABEL_ORDER,
    NOTCH_HALF_WIDTH_HZ,
    REST_LABEL,
    SAMPLE_RATE_HZ,
    WINDOW_SAMPLES,
    channel_rms,
    preprocess_window,
)


ALL_LABELS = (*GESTURE_LABEL_ORDER, REST_LABEL)
ACTION_TRIM_START_SAMPLES = 250
ACTION_TRIM_END_SAMPLES = 150
MIN_ACTION_SAMPLES_AFTER_TRIM = 1_500
MAX_RAIL_FRACTION = 0.01
RAIL_LOW_ADC = 5
RAIL_HIGH_ADC = 610

REQUIRED_COLUMNS = (
    "sample_index",
    "a0_adc",
    "a1_adc",
    "a2_adc",
    "phase",
    "target_label",
    "trial_id",
)


@dataclass
class DatasetBundle:
    records: list[dict[str, Any]]
    quality: list[dict[str, Any]]
    sessions: list[dict[str, Any]]
    include_all_dates: bool = False


def _missing_samples(indices: np.ndarray) -> tuple[int, int]:
    if len(indices) < 2:
        return 0, 0
    previous = indices[:-1].astype(np.uint64)
    current = indices[1:].astype(np.uint64)
    delta = (current - previous) & np.uint64(0xFFFFFFFF)
    forward = (delta > 1) & (delta < np.uint64(0x80000000))
    missing = int(np.sum(delta[forward] - 1, dtype=np.uint64))
    out_of_order = int(np.sum((delta == 0) | (delta >= np.uint64(0x80000000))))
    return missing, out_of_order


def _middle_window(signals: np.ndarray) -> np.ndarray:
    start = max(0, (len(signals) - WINDOW_SAMPLES) // 2)
    return signals[start:start + WINDOW_SAMPLES]


def _signal_summary(signals: np.ndarray) -> dict[str, Any]:
    raw_std = np.std(signals, axis=0)
    raw_range = np.ptp(signals, axis=0)
    rail_fraction = np.mean(
        (signals <= RAIL_LOW_ADC) | (signals >= RAIL_HIGH_ADC), axis=0
    )
    filtered = preprocess_window(
        _middle_window(signals),
        low_hz=FILTER_LOW_HZ,
        high_hz=FINAL_FILTER_HIGH_HZ,
        notch_frequencies_hz=FINAL_NOTCH_FREQUENCIES_HZ,
        notch_half_width_hz=NOTCH_HALF_WIDTH_HZ,
    )
    rms = channel_rms(filtered)
    correlations = np.corrcoef(signals.T)
    max_duplicate_correlation = float(
        np.max(np.abs(correlations[np.triu_indices(3, k=1)]))
    )
    return {
        "raw_std_adc": raw_std.tolist(),
        "raw_range_adc": raw_range.tolist(),
        "filtered_rms_adc": rms.tolist(),
        "rail_fraction": rail_fraction.tolist(),
        "max_abs_channel_correlation": max_duplicate_correlation,
    }


def _quality_reason(
    signals: np.ndarray,
    missing: int,
    out_of_order: int,
    minimum_samples: int = MIN_ACTION_SAMPLES_AFTER_TRIM,
) -> str:
    if len(signals) < minimum_samples:
        return f"too short after trim: {len(signals)} samples"
    if not np.isfinite(signals).all():
        return "non-finite ADC value"
    if missing:
        return f"{missing} missing sample groups inside ACTION"
    if out_of_order:
        return f"{out_of_order} duplicate/out-of-order samples inside ACTION"
    rail_fraction = np.mean(
        (signals <= RAIL_LOW_ADC) | (signals >= RAIL_HIGH_ADC), axis=0
    )
    if float(np.max(rail_fraction)) > MAX_RAIL_FRACTION:
        return f"ADC rail fraction above {MAX_RAIL_FRACTION:.1%}"
    raw_range = np.ptp(signals, axis=0)
    if bool(np.any(raw_range < 1.0)):
        return "one or more channels are effectively constant"
    return ""


def load_guided_sessions(
    data_dir: Path,
    *,
    include_all_dates: bool = False,
) -> DatasetBundle:
    records: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    allowed = set(ALL_LABELS)
    data_dir.mkdir(parents=True, exist_ok=True)

    dated_directories = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir()
        and _is_date_directory_name(path.name)
    )
    scan_directories = (
        dated_directories
        if include_all_dates and dated_directories
        else [dated_directories[-1] if dated_directories else data_dir]
    )
    paths = sorted(
        path
        for directory in scan_directories
        for path in directory.rglob("RAW_EMG_*.csv")
    )
    for path in paths:
        relative_parts = set(path.relative_to(data_dir).parts[:-1])
        if (
            "통합파일_보관" in relative_parts
            or "미완료" in relative_parts
            or "결함_제외" in relative_parts
            or "저장실패" in relative_parts
        ):
            continue
        session_id = path.stem
        json_path = path.with_suffix(".json")
        session_entry: dict[str, Any] = {
            "session_id": session_id,
            "file": path.name,
            "status": "excluded",
            "reason": "",
            "trials": {},
        }
        if not json_path.exists():
            if "무작위" in path.name:
                session_entry["reason"] = "recording is incomplete or still open (JSON missing)"
                sessions.append(session_entry)
            continue
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            session_entry["reason"] = f"metadata read error: {exc}"
            sessions.append(session_entry)
            continue
        mode = str(metadata.get("mode", ""))
        if mode not in {"manual", "guided", "guided_trial"}:
            continue
        if mode == "guided_trial" and metadata.get("protocol_trial_completed") is not True:
            session_entry["reason"] = "guided trial was stopped before RECOVERY completed"
            sessions.append(session_entry)
            continue
        participant_id = str(metadata.get("participant_id") or "").strip()
        arm_side = str(metadata.get("arm_side") or "").strip()
        attachment_id = str(metadata.get("electrode_attachment_id") or "").strip()
        if attachment_id:
            # An attachment is the scientifically meaningful grouping unit:
            # files collected without moving electrodes must never leak across splits.
            session_id = "::".join(
                value or "미기록" for value in (participant_id, arm_side, attachment_id)
            )
        elif mode == "guided_trial":
            session_id = str(metadata.get("protocol_session_id") or path.stem)
        else:
            session_id = str(
                metadata.get("recording_session_id")
                or f"manual_{str(metadata.get('created_at', 'unknown'))[:10]}"
            )
        session_entry["session_id"] = session_id
        if int(metadata.get("sample_rate_hz", 0)) != SAMPLE_RATE_HZ:
            session_entry["reason"] = (
                f"sample rate is not {SAMPLE_RATE_HZ} Hz: "
                f"{metadata.get('sample_rate_hz')}"
            )
            sessions.append(session_entry)
            continue
        try:
            frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
        except Exception as exc:
            session_entry["reason"] = f"CSV read error: {exc}"
            sessions.append(session_entry)
            continue

        frame["phase"] = frame["phase"].astype(str).str.strip()
        frame["target_label"] = frame["target_label"].astype(str).str.strip()
        action = (
            frame.copy()
            if mode == "manual"
            else frame[frame["phase"].eq("ACTION")].copy()
        )
        if action.empty:
            session_entry["reason"] = "no ACTION rows"
            sessions.append(session_entry)
            continue

        session_counts: Counter[str] = Counter()
        for trial_value, group in action.groupby("trial_id", sort=True):
            labels = [value for value in group["target_label"].unique() if value in allowed]
            base_entry: dict[str, Any] = {
                "session_id": session_id,
                "file": path.name,
                "trial_number": str(trial_value),
                "status": "excluded",
                "reason": "",
            }
            if len(labels) != 1:
                base_entry["reason"] = f"unsupported/ambiguous label: {labels}"
                quality.append(base_entry)
                continue
            label = labels[0]
            numeric = group.loc[:, ["sample_index", "a0_adc", "a1_adc", "a2_adc"]].apply(
                pd.to_numeric, errors="coerce"
            )
            indices_all = numeric["sample_index"].to_numpy(np.uint64)
            signals_all = numeric[["a0_adc", "a1_adc", "a2_adc"]].to_numpy(np.float32)
            if mode == "manual":
                signals = signals_all
                indices = indices_all
            else:
                trim_end = len(signals_all) - ACTION_TRIM_END_SAMPLES
                if trim_end <= ACTION_TRIM_START_SAMPLES:
                    signals = signals_all[:0]
                    indices = indices_all[:0]
                else:
                    signals = signals_all[ACTION_TRIM_START_SAMPLES:trim_end]
                    indices = indices_all[ACTION_TRIM_START_SAMPLES:trim_end]
            missing, out_of_order = _missing_samples(indices)
            reason = _quality_reason(
                signals,
                missing,
                out_of_order,
                minimum_samples=WINDOW_SAMPLES if mode == "manual" else MIN_ACTION_SAMPLES_AFTER_TRIM,
            )
            trial_id = (
                path.stem
                if mode == "manual"
                else f"{session_id}::trial_{trial_value}"
            )
            base_entry.update(
                {
                    "trial_id": trial_id,
                    "label": label,
                    "raw_action_rows": int(len(signals_all)),
                    "used_rows": int(len(signals)),
                    "missing_samples": missing,
                    "out_of_order_samples": out_of_order,
                }
            )
            if len(signals) >= WINDOW_SAMPLES and np.isfinite(signals).all():
                base_entry.update(_signal_summary(signals))
            if reason:
                base_entry["reason"] = reason
                quality.append(base_entry)
                continue
            base_entry["status"] = "included"
            quality.append(base_entry)
            session_counts[label] += 1
            records.append(
                {
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "trial_number": str(trial_value),
                    "file": path.name,
                    "label": label,
                    "created_at": metadata.get("created_at"),
                    "mode": mode,
                    "signals": signals,
                }
            )

        session_entry.update(
            {
                "status": "included" if sum(session_counts.values()) else "excluded",
                "reason": "" if sum(session_counts.values()) else "no valid ACTION trials",
                "created_at": metadata.get("created_at"),
                "stopped_at": metadata.get("stopped_at"),
                "missing_sample_groups_file": int(metadata.get("missing_sample_groups", 0)),
                "trials": dict(session_counts),
            }
        )
        sessions.append(session_entry)

    consolidated: dict[str, dict[str, Any]] = {}
    excluded_sessions: list[dict[str, Any]] = []
    for entry in sessions:
        if entry.get("status") != "included":
            excluded_sessions.append(entry)
            continue
        session_id = str(entry["session_id"])
        if session_id not in consolidated:
            consolidated[session_id] = {
                "session_id": session_id,
                "file": [],
                "status": "included",
                "reason": "",
                "created_at": entry.get("created_at"),
                "stopped_at": entry.get("stopped_at"),
                "missing_sample_groups_file": 0,
                "trials": {label: 0 for label in ALL_LABELS},
            }
        target = consolidated[session_id]
        files = target["file"]
        if isinstance(files, list):
            files.append(str(entry["file"]))
        target["missing_sample_groups_file"] = int(
            target["missing_sample_groups_file"]
        ) + int(entry.get("missing_sample_groups_file", 0))
        for label, count in dict(entry.get("trials", {})).items():
            target["trials"][label] = int(target["trials"].get(label, 0)) + int(count)
        if str(entry.get("created_at") or "") < str(target.get("created_at") or "~"):
            target["created_at"] = entry.get("created_at")
        if str(entry.get("stopped_at") or "") > str(target.get("stopped_at") or ""):
            target["stopped_at"] = entry.get("stopped_at")
    return DatasetBundle(
        records=records,
        quality=quality,
        sessions=[*consolidated.values(), *excluded_sessions],
        include_all_dates=include_all_dates,
    )


def _is_date_directory_name(value: str) -> bool:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        return False


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    included_sessions = [entry for entry in bundle.sessions if entry["status"] == "included"]
    counts = Counter(str(record["label"]) for record in bundle.records)
    per_session: dict[str, dict[str, int]] = defaultdict(dict)
    for session in included_sessions:
        per_session[str(session["session_id"])] = {
            label: int(dict(session["trials"]).get(label, 0)) for label in ALL_LABELS
        }

    warnings: list[str] = []
    blockers: list[str] = []
    if not included_sessions:
        blockers.append("완료된 측정 데이터가 없습니다.")
    for label in ALL_LABELS:
        if counts[label] < 10:
            blockers.append(f"{label}: 유효 측정 {counts[label]}회 (최소 10회 필요)")
        elif counts[label] < 30:
            warnings.append(f"{label}: {counts[label]}회로 적음 (권장 30회 이상)")
    if len(included_sessions) < 3:
        warnings.append(
            f"완료 세션 {len(included_sessions)}개: 새로 붙인 전극에 대한 시험을 위해 3세션 이상 권장"
        )

    rms_by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for entry in bundle.quality:
        if entry.get("status") == "included" and "filtered_rms_adc" in entry:
            rms_by_label[str(entry["label"])].append(
                np.asarray(entry["filtered_rms_adc"], dtype=np.float64)
            )
    median_rms = {
        label: np.median(np.stack(values), axis=0).tolist()
        for label, values in rms_by_label.items()
        if values
    }
    if REST_LABEL in median_rms:
        rest = np.maximum(np.asarray(median_rms[REST_LABEL]), 0.05)
        for label in GESTURE_LABEL_ORDER:
            if label not in median_rms:
                continue
            ratio = np.asarray(median_rms[label]) / rest
            if float(np.max(ratio)) < 1.35:
                warnings.append(
                    f"{label}: 휴식 대비 최대 RMS 비율 {float(np.max(ratio)):.2f}로 활성 구분이 약함"
                )

    return {
        "ready_to_train": not blockers,
        "completed_guided_sessions": len(included_sessions),
        "trial_counts": {label: int(counts[label]) for label in ALL_LABELS},
        "per_session_counts": dict(per_session),
        "median_filtered_rms_adc": median_rms,
        "included_trials": len(bundle.records),
        "excluded_trials": int(sum(entry.get("status") == "excluded" for entry in bundle.quality)),
        "ignored_or_incomplete_sessions": int(
            sum(entry.get("status") == "excluded" for entry in bundle.sessions)
        ),
        "warnings": warnings,
        "blockers": blockers,
        "policy": {
            "input": (
                "all dated folders: valid manual and guided recordings"
                if bundle.include_all_dates
                else "latest date folder: valid manual and guided recordings"
            ),
            "used_phase": "manual full record; guided ACTION only",
            "trim_samples": {
                "start": ACTION_TRIM_START_SAMPLES,
                "end": ACTION_TRIM_END_SAMPLES,
            },
            "older_date_folders": (
                "included" if bundle.include_all_dates else "ignored"
            ),
        },
    }


def write_preflight_reports(bundle: DatasetBundle, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(bundle)
    (output_dir / "dataset_preflight.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    serializable_quality = []
    for entry in bundle.quality:
        serializable_quality.append(
            {key: value for key, value in entry.items() if key != "signals"}
        )
    (output_dir / "trial_quality.json").write_text(
        json.dumps(serializable_quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "session_quality.json").write_text(
        json.dumps(bundle.sessions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "퍼플팀 EMG 최종 데이터 사전 검사",
        "",
        f"포함 세션: {summary['completed_guided_sessions']}개",
        f"유효 시행: {summary['included_trials']}회",
        *[
            f"- {label}: {summary['trial_counts'][label]}회"
            for label in ALL_LABELS
        ],
        "",
        "학습 가능: " + ("예" if summary["ready_to_train"] else "아니오"),
    ]
    if summary["blockers"]:
        lines.extend(["", "해결해야 함:", *[f"- {v}" for v in summary["blockers"]]])
    if summary["warnings"]:
        lines.extend(["", "주의:", *[f"- {v}" for v in summary["warnings"]]])
    lines.extend(
        [
            "",
            (
                "중요: 날짜 폴더 전체를 사용합니다. 전극 위치가 다른 세션의 domain shift를 확인하세요."
                if bundle.include_all_dates
                else "중요: 가장 최근 날짜 폴더만 사용하여 예전 전극 위치 데이터를 제외합니다."
            ),
            "수동 측정은 전체 기록, 무작위 측정은 ACTION 구간만 사용합니다.",
        ]
    )
    (output_dir / "dataset_preflight.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8-sig"
    )
    return summary
