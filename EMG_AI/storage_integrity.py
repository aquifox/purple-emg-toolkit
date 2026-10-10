"""Audit and repair the Purple Team dated RAW EMG storage tree."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_EXTENSIONS = (".bin", ".csv", ".json")
QUARANTINE_NAMES = {"미완료", "결함_제외", "저장실패", "통합파일_보관"}
PACKET_SIZE_BYTES = 14
REQUIRED_CSV_COLUMNS = {
    "sample_index",
    "elapsed_seconds",
    "a0_adc",
    "a1_adc",
    "a2_adc",
    "phase",
    "target_label",
    "trial_id",
}


def is_date_name(name: str) -> bool:
    try:
        return datetime.strptime(name, "%Y-%m-%d").strftime("%Y-%m-%d") == name
    except ValueError:
        return False


def _issue(level: str, code: str, path: Path, detail: str) -> dict[str, str]:
    return {"level": level, "code": code, "path": str(path), "detail": detail}


def _is_quarantined(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return True
    return any(part in QUARANTINE_NAMES for part in parts)


def scan_storage(records_root: Path, *, deep: bool = False) -> dict[str, Any]:
    records_root.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, str]] = []
    label_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    partial_files = sorted(records_root.rglob("*.part"))
    for path in partial_files:
        issues.append(
            _issue(
                "error",
                "partial_file",
                path,
                "기록이 정상 종료되지 않은 임시파일입니다. 학습에는 사용되지 않습니다.",
            )
        )

    stems: dict[Path, set[str]] = {}
    for path in records_root.rglob("RAW_EMG*"):
        if not path.is_file() or path.name.endswith(".part"):
            continue
        suffix = path.suffix.lower()
        if suffix not in REQUIRED_EXTENSIONS:
            continue
        stems.setdefault(path.with_suffix(""), set()).add(suffix)

    active_trials = 0
    quarantined_trials = 0
    total_rows = 0
    for stem, present in sorted(stems.items(), key=lambda item: str(item[0])):
        missing = sorted(set(REQUIRED_EXTENSIONS) - present)
        if missing:
            issues.append(
                _issue(
                    "error",
                    "missing_companion",
                    stem,
                    "짝 파일 누락: " + ", ".join(missing),
                )
            )
            continue

        json_path = stem.with_suffix(".json")
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            issues.append(_issue("error", "invalid_json", json_path, str(exc)))
            continue

        expected_rows = int(metadata.get("rows") or 0)
        total_rows += expected_rows
        bin_path = stem.with_suffix(".bin")
        bin_size = bin_path.stat().st_size
        if bin_size % PACKET_SIZE_BYTES:
            issues.append(
                _issue(
                    "error",
                    "binary_remainder",
                    bin_path,
                    f"BIN 크기가 {PACKET_SIZE_BYTES}바이트 패킷 단위가 아닙니다.",
                )
            )
        binary_rows = bin_size // PACKET_SIZE_BYTES
        if binary_rows != expected_rows:
            issues.append(
                _issue(
                    "error",
                    "row_count_mismatch",
                    stem,
                    f"JSON rows={expected_rows}, BIN packets={binary_rows}",
                )
            )

        if deep:
            csv_path = stem.with_suffix(".csv")
            try:
                with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
                    reader = csv.reader(source)
                    header = next(reader)
                    csv_rows = sum(1 for _ in reader)
                missing_columns = sorted(REQUIRED_CSV_COLUMNS - set(header))
                if missing_columns:
                    issues.append(
                        _issue(
                            "error",
                            "csv_columns_missing",
                            csv_path,
                            "필수 열 누락: " + ", ".join(missing_columns),
                        )
                    )
                if csv_rows != expected_rows:
                    issues.append(
                        _issue(
                            "error",
                            "row_count_mismatch",
                            csv_path,
                            f"JSON rows={expected_rows}, CSV rows={csv_rows}",
                        )
                    )
            except Exception as exc:
                issues.append(_issue("error", "csv_read_error", csv_path, str(exc)))

        relative = json_path.relative_to(records_root)
        created_date = str(metadata.get("created_at") or "")[:10]
        top_folder = relative.parts[0] if len(relative.parts) > 1 else ""
        quarantined = _is_quarantined(json_path, records_root)
        if quarantined:
            quarantined_trials += 1
            continue
        if not top_folder:
            issues.append(
                _issue(
                    "error",
                    "misplaced_root_trial",
                    json_path,
                    f"날짜 폴더 밖에 있습니다. 메타데이터 날짜: {created_date or '없음'}",
                )
            )
            continue
        if not is_date_name(top_folder):
            issues.append(
                _issue("error", "invalid_date_folder", json_path, f"상위 폴더: {top_folder}")
            )
            continue
        if created_date and created_date != top_folder:
            issues.append(
                _issue(
                    "error",
                    "date_mismatch",
                    json_path,
                    f"폴더={top_folder}, created_at={created_date}",
                )
            )
        active_trials += 1
        date_counts[top_folder] += 1
        mode_counts[str(metadata.get("mode") or "unknown")] += 1

        label = ""
        for event in metadata.get("events", []):
            candidate = str(event.get("target_label") or "").strip()
            if candidate:
                label = candidate
        if not label:
            name = stem.name.removeprefix("RAW_EMG_")
            label = name.split("_무작위", 1)[0]
            label = re.sub(r"_\d{8}_\d{6}_\d{3}(?:_.+)?$", "", label)
        label_counts[label or "unknown"] += 1

    error_count = sum(issue["level"] == "error" for issue in issues)
    return {
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "records_root": str(records_root.resolve()),
        "healthy": error_count == 0,
        "deep_scan": deep,
        "active_trials": active_trials,
        "quarantined_trials": quarantined_trials,
        "total_rows": total_rows,
        "duration_minutes_at_1khz": total_rows / 60_000.0,
        "date_counts": dict(sorted(date_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "partial_files": len(partial_files),
        "error_count": error_count,
        "issues": issues,
    }


def repair_misplaced_root_trials(records_root: Path) -> list[dict[str, str]]:
    """Move complete root-level triplets into their metadata date folder."""
    actions: list[dict[str, str]] = []
    records_root.mkdir(parents=True, exist_ok=True)
    for json_path in sorted(records_root.glob("RAW_EMG*.json")):
        metadata = json.loads(json_path.read_text(encoding="utf-8-sig"))
        date_name = str(metadata.get("created_at") or "")[:10]
        if not is_date_name(date_name):
            raise ValueError(f"올바른 created_at 날짜가 없습니다: {json_path}")
        source_stem = json_path.with_suffix("")
        sources = [source_stem.with_suffix(suffix) for suffix in REQUIRED_EXTENSIONS]
        if not all(path.exists() for path in sources):
            raise FileNotFoundError(f"짝 파일이 누락되어 이동할 수 없습니다: {source_stem}")
        target_dir = records_root / date_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_stem = target_dir / source_stem.name
        if any(target_stem.with_suffix(suffix).exists() for suffix in REQUIRED_EXTENSIONS):
            for number in range(1, 10_000):
                candidate = target_dir / f"{source_stem.name}_복구{number:02d}"
                if not any(candidate.with_suffix(suffix).exists() for suffix in REQUIRED_EXTENSIONS):
                    target_stem = candidate
                    break
            else:
                raise OSError(f"중복 없는 복구 파일명을 만들 수 없습니다: {source_stem}")
        for source, suffix in zip(sources, REQUIRED_EXTENSIONS):
            shutil.move(str(source), str(target_stem.with_suffix(suffix)))
        actions.append({"from": str(source_stem), "to": str(target_stem)})
    return actions


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        "퍼플팀 EMG RAW 저장소 검사",
        f"상태: {'정상' if report['healthy'] else '오류 있음'}",
        f"활성 기록: {report['active_trials']}회",
        f"격리 기록: {report['quarantined_trials']}회",
        f"임시파일: {report['partial_files']}개",
        "날짜별: " + ", ".join(f"{k} {v}회" for k, v in report["date_counts"].items()),
        "동작별: " + ", ".join(f"{k} {v}회" for k, v in report["label_counts"].items()),
    ]
    if report["issues"]:
        lines.append("문제:")
        lines.extend(
            f"- [{item['code']}] {item['path']}: {item['detail']}"
            for item in report["issues"]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_root", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent / "raw_records")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--repair-root", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repair_root:
        repair_misplaced_root_trials(args.records_root)
    report = scan_storage(args.records_root, deep=args.deep)
    print(format_summary(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if report["healthy"] else 1)


if __name__ == "__main__":
    main()
