"""One-command preflight check for the final three-finger dataset."""

from __future__ import annotations

from pathlib import Path

from final_dataset import ALL_LABELS, load_guided_sessions, write_preflight_reports


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "raw_records"
OUTPUT_DIR = SCRIPT_DIR / "outputs_final_5class"


def main() -> int:
    bundle = load_guided_sessions(DATA_DIR)
    summary = write_preflight_reports(bundle, OUTPUT_DIR)
    print("\n=== FINAL EMG DATA PREFLIGHT ===")
    print(f"Completed guided sessions: {summary['completed_guided_sessions']}")
    for label in ALL_LABELS:
        print(f"  {label}: {summary['trial_counts'][label]} trials")
    if summary["warnings"]:
        print("\nWARNINGS")
        for warning in summary["warnings"]:
            print(f"  - {warning}")
    if summary["blockers"]:
        print("\nNOT READY")
        for blocker in summary["blockers"]:
            print(f"  - {blocker}")
        print(f"\nReport: {OUTPUT_DIR / 'dataset_preflight.txt'}")
        return 2
    print("\nREADY TO TRAIN")
    print(f"Report: {OUTPUT_DIR / 'dataset_preflight.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
