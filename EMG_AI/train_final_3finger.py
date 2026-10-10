"""Train the final rest + 4-gesture EMG RF and 1D-CNN models.

Pipeline
--------
1. Read valid manual trials and completed randomized-protocol trials.
2. Restore each file/trial_id as one independent trial.
3. Split by whole session when at least three sessions exist; otherwise use
   whole-trial fallback splits and state that limitation in the report.
4. Learn a rest/active RMS-ratio gate.
5. Classify active windows as index/middle/ring/fist using RF and a compact 1D CNN.
6. Evaluate at the whole-trial level and retrain deployment models on all data.
"""

from __future__ import annotations

import copy
import argparse
import json
import platform
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from final_dataset import (
    ALL_LABELS,
    DatasetBundle,
    load_guided_sessions,
    write_preflight_reports,
)
from raw_cnn_model import RawEmg1DCNN
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
    extract_rf_features,
    feature_names,
    normalize_for_classifier,
    preprocess_window,
)


SEED = 20260902
WINDOW_STRIDE = 250
RF_FEATURE_VERSION = "final_v2"
MAX_EPOCHS = 120
PATIENCE = 16
MIN_CONFIDENCE = 0.34
MAX_CONFIDENCE = 0.90

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "raw_records"
OUTPUT_DIR = SCRIPT_DIR / "outputs_final_5class"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _trial_split_for_records(
    records: list[dict[str, Any]],
    *,
    train_fraction: float = 0.60,
    include_test: bool = True,
) -> dict[str, set[str]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["label"])].append(
            (str(record.get("created_at") or record["file"]), str(record["trial_id"]))
        )
    result = {"train": set(), "validation": set(), "test": set()}
    for label, dated_ids in sorted(grouped.items()):
        ids = [trial_id for _created, trial_id in sorted(dated_ids)]
        count = len(ids)
        if include_test:
            validation_count = max(1, int(round(count * 0.20)))
            test_count = max(1, int(round(count * 0.20)))
            if count - validation_count - test_count < 2:
                validation_count = 1
                test_count = 1
            train_count = count - validation_count - test_count
            result["train"].update(ids[:train_count])
            result["validation"].update(ids[train_count:train_count + validation_count])
            result["test"].update(ids[train_count + validation_count:])
        else:
            train_count = max(2, int(round(count * train_fraction)))
            train_count = min(train_count, count - 1)
            result["train"].update(ids[:train_count])
            result["validation"].update(ids[train_count:])
    return result


def _labels_in_ids(records: list[dict[str, Any]], ids: set[str]) -> set[str]:
    return {
        str(record["label"])
        for record in records
        if str(record["trial_id"]) in ids
    }


def split_records(records: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    sessions = sorted({str(record["session_id"]) for record in records})
    all_labels = set(ALL_LABELS)
    if len(sessions) >= 3:
        test_count = max(1, int(round(len(sessions) * 0.20)))
        validation_count = max(1, int(round(len(sessions) * 0.20)))
        if len(sessions) - test_count - validation_count < 1:
            test_count = validation_count = 1
        session_split = {
            "train": set(sessions[: len(sessions) - validation_count - test_count]),
            "validation": set(sessions[len(sessions) - validation_count - test_count: -test_count]),
            "test": set(sessions[-test_count:]),
        }
        trial_split = {
            name: {
                str(record["trial_id"])
                for record in records
                if str(record["session_id"]) in selected
            }
            for name, selected in session_split.items()
        }
        if all(_labels_in_ids(records, ids) == all_labels for ids in trial_split.values()):
            return trial_split, {
                "mode": "session_holdout",
                "reliability": "preferred: electrode/session changes are held out",
                "sessions": {name: sorted(values) for name, values in session_split.items()},
            }

    if len(sessions) == 2:
        training_session, test_session = sessions
        training_records = [
            record for record in records if str(record["session_id"]) == training_session
        ]
        inner = _trial_split_for_records(training_records, train_fraction=0.80, include_test=False)
        split = {
            "train": inner["train"],
            "validation": inner["validation"],
            "test": {
                str(record["trial_id"])
                for record in records
                if str(record["session_id"]) == test_session
            },
        }
        if all(_labels_in_ids(records, ids) == all_labels for ids in split.values()):
            return split, {
                "mode": "two_session_holdout",
                "reliability": "test session held out; validation is same-electrode trial holdout",
                "sessions": {
                    "train_validation": [training_session],
                    "test": [test_session],
                },
            }

    split = _trial_split_for_records(records, include_test=True)
    return split, {
        "mode": "chronological_trial_holdout_same_date",
        "reliability": (
            "provisional: later trials are held out, but a new electrode attachment is not tested"
        ),
        "sessions": {"all": sessions},
    }


def build_windows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for record in records:
        signals = np.asarray(record["signals"], dtype=np.float32)
        starts = list(range(0, len(signals) - WINDOW_SAMPLES + 1, WINDOW_STRIDE))
        if len(starts) > 5:
            starts = sorted(
                {
                    starts[index]
                    for index in np.linspace(0, len(starts) - 1, 5).round().astype(int)
                }
            )
        for start in starts:
            filtered = preprocess_window(
                signals[start:start + WINDOW_SAMPLES],
                low_hz=FILTER_LOW_HZ,
                high_hz=FINAL_FILTER_HIGH_HZ,
                notch_frequencies_hz=FINAL_NOTCH_FREQUENCIES_HZ,
                notch_half_width_hz=NOTCH_HALF_WIDTH_HZ,
            )
            windows.append(
                {
                    "session_id": str(record["session_id"]),
                    "trial_id": str(record["trial_id"]),
                    "label": str(record["label"]),
                    "start": int(start),
                    "filtered": filtered,
                    "classified": normalize_for_classifier(filtered),
                    "rms": channel_rms(filtered),
                }
            )
    return windows


def select_windows(windows: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    return [window for window in windows if str(window["trial_id"]) in ids]


def session_baselines(windows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rest_rms = [
        np.asarray(window["rms"], dtype=np.float32)
        for window in windows
        if str(window["label"]) == REST_LABEL
    ]
    if not rest_rms:
        raise RuntimeError("No rest windows are available for the activation gate.")
    global_baseline = np.median(np.stack(rest_rms), axis=0).astype(np.float32)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for window in windows:
        if str(window["label"]) == REST_LABEL:
            grouped[str(window["session_id"])].append(
                np.asarray(window["rms"], dtype=np.float32)
            )
    return {
        session: np.median(np.stack(values), axis=0).astype(np.float32)
        for session, values in grouped.items()
    }, global_baseline


def activation_scores(
    windows: list[dict[str, Any]],
    baselines: dict[str, np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            float(
                np.max(
                    np.asarray(window["rms"], dtype=np.float32)
                    / np.maximum(baselines.get(str(window["session_id"]), fallback), 0.05)
                )
            )
            for window in windows
        ],
        dtype=np.float32,
    )


def choose_activation_threshold(
    windows: list[dict[str, Any]],
    baselines: dict[str, np.ndarray],
    fallback: np.ndarray,
) -> tuple[float, dict[str, float]]:
    scores = activation_scores(windows, baselines, fallback)
    actual = np.asarray([str(window["label"]) != REST_LABEL for window in windows])
    if not np.any(actual) or not np.any(~actual):
        raise RuntimeError("Activation threshold selection needs both rest and gesture windows.")
    upper = max(2.0, float(np.percentile(scores, 99.5)))
    candidates = np.linspace(1.05, upper, 800)
    labels = np.asarray([str(window["label"]) for window in windows])
    gesture_labels = sorted(set(labels[actual]))
    best: tuple[float, float, float, float, float] | None = None
    best_threshold = 1.5
    for threshold in candidates:
        predicted = scores >= threshold
        per_label_sensitivity = [
            float(np.mean(predicted[labels == label])) for label in gesture_labels
        ]
        sensitivity = float(np.mean(per_label_sensitivity))
        minimum_sensitivity = float(min(per_label_sensitivity))
        specificity = float(np.mean(~predicted[~actual]))
        balanced = 0.5 * (sensitivity + specificity)
        meets_gesture_floor = minimum_sensitivity >= 0.85
        candidate = (
            float(meets_gesture_floor),
            specificity if meets_gesture_floor else minimum_sensitivity,
            balanced,
            sensitivity,
            float(threshold),
        )
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    predicted = scores >= best_threshold
    per_label = {
        label: float(np.mean(predicted[labels == label])) for label in gesture_labels
    }
    sensitivity = float(np.mean(list(per_label.values())))
    specificity = float(np.mean(~predicted[~actual]))
    return best_threshold, {
        "sensitivity": sensitivity,
        "minimum_gesture_sensitivity": float(min(per_label.values())),
        "per_gesture_sensitivity": per_label,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
    }


def gate_metrics(
    windows: list[dict[str, Any]],
    baselines: dict[str, np.ndarray],
    fallback: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    scores = activation_scores(windows, baselines, fallback)
    actual = np.asarray([str(window["label"]) != REST_LABEL for window in windows])
    predicted = scores >= threshold
    sensitivity = float(np.mean(predicted[actual])) if np.any(actual) else 0.0
    specificity = float(np.mean(~predicted[~actual])) if np.any(~actual) else 0.0
    return {
        "windows": len(windows),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "false_activation_rate": 1.0 - specificity,
        "false_positive_windows": int(np.sum(predicted & ~actual)),
        "false_negative_windows": int(np.sum(~predicted & actual)),
    }


def active_gesture_windows(
    windows: list[dict[str, Any]],
    baselines: dict[str, np.ndarray],
    fallback: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    scores = activation_scores(windows, baselines, fallback)
    return [
        window
        for window, score in zip(windows, scores)
        if str(window["label"]) != REST_LABEL and score >= threshold
    ]


def all_gesture_windows(
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classifier training must not discard weak gestures selected by the gate."""
    return [window for window in windows if str(window["label"]) != REST_LABEL]


def rf_arrays(
    windows: list[dict[str, Any]], label_to_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not windows:
        raise RuntimeError("No active windows survived the gate for RF.")
    x = np.stack(
        [
            extract_rf_features(
                np.asarray(window["classified"], dtype=np.float32), RF_FEATURE_VERSION
            )
            for window in windows
        ]
    )
    y = np.asarray([label_to_index[str(window["label"])] for window in windows])
    groups = [str(window["trial_id"]) for window in windows]
    return x, y, groups


def cnn_arrays(
    windows: list[dict[str, Any]], label_to_index: dict[str, int], scale: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not windows:
        raise RuntimeError("No active windows survived the gate for CNN.")
    x = np.stack(
        [np.asarray(window["classified"], dtype=np.float32).T for window in windows]
    )
    x = np.clip(x / max(scale, 1e-6), -6.0, 6.0).astype(np.float32)
    y = np.asarray([label_to_index[str(window["label"])] for window in windows], dtype=np.int64)
    groups = [str(window["trial_id"]) for window in windows]
    return x, y, groups


def _loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(32, max(1, len(x))),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def train_cnn_evaluation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    class_count: int,
) -> tuple[RawEmg1DCNN, int, list[dict[str, float]]]:
    model = RawEmg1DCNN(class_count)
    counts = np.bincount(y_train, minlength=class_count)
    class_weights = len(y_train) / (class_count * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-4)
    train_loader = _loader(x_train, y_train, True)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for inputs, targets in train_loader:
            global_gain = 0.80 + 0.40 * torch.rand((len(inputs), 1, 1))
            channel_gain = 0.90 + 0.20 * torch.rand((len(inputs), 3, 1))
            augmented = inputs * global_gain * channel_gain
            augmented += 0.012 * torch.randn_like(augmented)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(augmented), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_validation))
            validation_loss = float(
                criterion(logits, torch.from_numpy(y_validation)).item()
            )
            validation_accuracy = float(np.mean(logits.argmax(dim=1).numpy() == y_validation))
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, history


def train_cnn_deployment(
    x: np.ndarray, y: np.ndarray, class_count: int, epochs: int
) -> RawEmg1DCNN:
    seed_everything(SEED + 1)
    model = RawEmg1DCNN(class_count)
    counts = np.bincount(y, minlength=class_count)
    class_weights = len(y) / (class_count * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-4)
    loader = _loader(x, y, True)
    for _ in range(max(8, epochs)):
        model.train()
        for inputs, targets in loader:
            global_gain = 0.80 + 0.40 * torch.rand((len(inputs), 1, 1))
            channel_gain = 0.90 + 0.20 * torch.rand((len(inputs), 3, 1))
            augmented = inputs * global_gain * channel_gain
            augmented += 0.012 * torch.randn_like(augmented)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(augmented), targets)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_cnn(model: nn.Module, x: np.ndarray) -> np.ndarray:
    results: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 64):
            logits = model(torch.from_numpy(x[start:start + 64]))
            results.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(results, axis=0)


def choose_confidence_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    confidence = np.max(probabilities, axis=1)
    prediction = np.argmax(probabilities, axis=1)
    labels = list(range(probabilities.shape[1]))
    best: tuple[float, float, float] | None = None
    best_threshold = 0.55
    for threshold in np.linspace(MIN_CONFIDENCE, MAX_CONFIDENCE, 113):
        accepted_prediction = prediction.copy()
        accepted_prediction[confidence < threshold] = -1
        macro_f1 = float(
            f1_score(y, accepted_prediction, labels=labels, average="macro", zero_division=0)
        )
        coverage = float(np.mean(confidence >= threshold))
        objective = macro_f1 - max(0.0, 0.65 - coverage) * 0.20
        candidate = (objective, coverage, float(threshold))
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    groups: list[str],
    confidence_threshold: float,
    labels: list[str],
) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    accepted = prediction.copy()
    accepted[confidence < confidence_threshold] = -1
    trial_true: list[int] = []
    trial_prediction: list[int] = []
    trial_confidence: list[float] = []
    for group in sorted(set(groups)):
        selected = np.asarray([value == group for value in groups])
        mean_probability = np.mean(probabilities[selected], axis=0)
        confidence_value = float(np.max(mean_probability))
        prediction_value = int(np.argmax(mean_probability))
        trial_true.append(int(y[np.flatnonzero(selected)[0]]))
        trial_prediction.append(
            prediction_value if confidence_value >= confidence_threshold else -1
        )
        trial_confidence.append(confidence_value)
    precision, recall, f1, support = precision_recall_fscore_support(
        trial_true,
        trial_prediction,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    return {
        "window_argmax_accuracy": float(accuracy_score(y, prediction)),
        "window_macro_f1_with_unknown": float(
            f1_score(
                y,
                accepted,
                labels=list(range(len(labels))),
                average="macro",
                zero_division=0,
            )
        ),
        "window_coverage": float(np.mean(confidence >= confidence_threshold)),
        "trial_accuracy_with_unknown": float(accuracy_score(trial_true, trial_prediction)),
        "trial_macro_f1_with_unknown": float(
            f1_score(
                trial_true,
                trial_prediction,
                labels=list(range(len(labels))),
                average="macro",
                zero_division=0,
            )
        ),
        "trial_coverage": float(np.mean(np.asarray(trial_prediction) >= 0)),
        "confidence_threshold": confidence_threshold,
        "per_class_trials": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_trials_argmax": {
            "labels": labels,
            "matrix": confusion_matrix(
                trial_true,
                [max(0, value) for value in trial_prediction],
                labels=list(range(len(labels))),
            ).tolist(),
            "note": "unknown predictions are included in scores but omitted from this compact matrix",
        },
        "mean_trial_confidence": float(np.mean(trial_confidence)),
    }


def probability_for_windows_rf(
    model: RandomForestClassifier, windows: list[dict[str, Any]]
) -> np.ndarray:
    x = np.stack(
        [
            extract_rf_features(
                np.asarray(window["classified"], dtype=np.float32), RF_FEATURE_VERSION
            )
            for window in windows
        ]
    )
    return model.predict_proba(x)


def probability_for_windows_cnn(
    model: RawEmg1DCNN, windows: list[dict[str, Any]], scale: float
) -> np.ndarray:
    x = np.stack(
        [np.asarray(window["classified"], dtype=np.float32).T for window in windows]
    )
    x = np.clip(x / max(scale, 1e-6), -6.0, 6.0).astype(np.float32)
    return predict_cnn(model, x)


def end_to_end_trial_metrics(
    windows: list[dict[str, Any]],
    probabilities: np.ndarray,
    baselines: dict[str, np.ndarray],
    fallback: np.ndarray,
    activation_threshold: float,
    confidence_threshold: float,
    gesture_labels: list[str],
    model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = activation_scores(windows, baselines, fallback)
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        grouped_indices[str(window["trial_id"])].append(index)
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    rows: list[dict[str, Any]] = []
    unknown_label = "판정 보류"
    for trial_id in sorted(grouped_indices):
        selected = np.asarray(grouped_indices[trial_id], dtype=np.int64)
        label = str(windows[int(selected[0])]["label"])
        active_mask = scores[selected] >= activation_threshold
        active_fraction = float(np.mean(active_mask))
        if active_fraction < 0.40:
            predicted = REST_LABEL
            confidence = 1.0 - active_fraction
        else:
            mean_probability = np.mean(probabilities[selected[active_mask]], axis=0)
            best = int(np.argmax(mean_probability))
            confidence = float(mean_probability[best])
            predicted = gesture_labels[best] if confidence >= confidence_threshold else unknown_label
        true_labels.append(label)
        predicted_labels.append(predicted)
        rows.append(
            {
                "model": model_name,
                "trial_id": trial_id,
                "session_id": str(windows[int(selected[0])]["session_id"]),
                "true_label": label,
                "predicted_label": predicted,
                "correct": label == predicted,
                "active_window_fraction": active_fraction,
                "confidence": confidence,
            }
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(ALL_LABELS),
        zero_division=0,
    )
    matrix_labels = [*ALL_LABELS, unknown_label]
    return {
        "trial_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "trial_macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=list(ALL_LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "unknown_trials": int(sum(value == unknown_label for value in predicted_labels)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(ALL_LABELS)
        },
        "confusion": {
            "labels": matrix_labels,
            "matrix": confusion_matrix(
                true_labels, predicted_labels, labels=matrix_labels
            ).tolist(),
        },
    }, rows


def _assert_active_labels(
    windows: list[dict[str, Any]], label_to_index: dict[str, int], split_name: str
) -> None:
    found = {str(window["label"]) for window in windows}
    missing = set(label_to_index) - found
    if missing:
        raise RuntimeError(
            f"{split_name}: activation gate removed every window for labels {sorted(missing)}"
        )


def main() -> None:
    global DATA_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Include every dated 1 kHz RAW folder instead of the latest date only.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="RAW data root; defaults to the project's raw_records folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Separate output directory; relative paths are resolved under EMG_AI.",
    )
    args = parser.parse_args()
    if args.data_dir is not None:
        DATA_DIR = args.data_dir.resolve()
    if args.output_dir is not None:
        OUTPUT_DIR = (
            args.output_dir
            if args.output_dir.is_absolute()
            else SCRIPT_DIR / args.output_dir
        )
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[2/5] 날짜별 RAW 데이터를 읽고 품질을 검사합니다.", flush=True)
    bundle: DatasetBundle = load_guided_sessions(
        DATA_DIR, include_all_dates=args.all_dates
    )
    preflight = write_preflight_reports(bundle, OUTPUT_DIR)
    if not preflight["ready_to_train"]:
        details = "\n".join(f"- {value}" for value in preflight["blockers"])
        raise RuntimeError(f"Final dataset is not ready:\n{details}")
    records = bundle.records
    split, split_info = split_records(records)
    windows = build_windows(records)
    split_windows = {name: select_windows(windows, ids) for name, ids in split.items()}
    for name, values in split_windows.items():
        if not values:
            raise RuntimeError(f"{name} split has no windows")

    all_baselines, all_fallback = session_baselines(windows)
    threshold_source = split_windows["validation"]
    activation_on, gate_selection = choose_activation_threshold(
        threshold_source, all_baselines, all_fallback
    )
    activation_off = max(1.05, activation_on * 0.70)
    active = {
        name: all_gesture_windows(values)
        for name, values in split_windows.items()
    }
    label_to_index = {label: index for index, label in enumerate(GESTURE_LABEL_ORDER)}
    for name, values in active.items():
        _assert_active_labels(values, label_to_index, name)

    x_rf_train, y_rf_train, _ = rf_arrays(active["train"], label_to_index)
    x_rf_validation, y_rf_validation, rf_validation_groups = rf_arrays(
        active["validation"], label_to_index
    )
    x_rf_test, y_rf_test, rf_test_groups = rf_arrays(active["test"], label_to_index)
    print(
        f"[데이터] 유효 기록 {len(records)}회 · 학습 창 {len(active['train'])}개",
        flush=True,
    )
    print("[3/5] Random Forest를 학습하고 시험합니다.", flush=True)
    rf_evaluation = RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_evaluation.fit(x_rf_train, y_rf_train)
    rf_validation_probability = rf_evaluation.predict_proba(x_rf_validation)
    rf_confidence = choose_confidence_threshold(y_rf_validation, rf_validation_probability)
    rf_test_probability = rf_evaluation.predict_proba(x_rf_test)
    rf_metrics = classification_metrics(
        y_rf_test,
        rf_test_probability,
        rf_test_groups,
        rf_confidence,
        list(GESTURE_LABEL_ORDER),
    )

    train_values = np.concatenate(
        [np.abs(np.asarray(window["classified"])).reshape(-1) for window in active["train"]]
    )
    cnn_scale = max(0.1, float(np.percentile(train_values, 99.5)))
    x_cnn_train, y_cnn_train, _ = cnn_arrays(active["train"], label_to_index, cnn_scale)
    x_cnn_validation, y_cnn_validation, cnn_validation_groups = cnn_arrays(
        active["validation"], label_to_index, cnn_scale
    )
    x_cnn_test, y_cnn_test, cnn_test_groups = cnn_arrays(
        active["test"], label_to_index, cnn_scale
    )
    print("[4/5] 1D CNN을 학습하고 시험합니다.", flush=True)
    cnn_evaluation, best_epoch, history = train_cnn_evaluation(
        x_cnn_train,
        y_cnn_train,
        x_cnn_validation,
        y_cnn_validation,
        len(GESTURE_LABEL_ORDER),
    )
    cnn_validation_probability = predict_cnn(cnn_evaluation, x_cnn_validation)
    cnn_confidence = choose_confidence_threshold(y_cnn_validation, cnn_validation_probability)
    cnn_test_probability = predict_cnn(cnn_evaluation, x_cnn_test)
    cnn_metrics = classification_metrics(
        y_cnn_test,
        cnn_test_probability,
        cnn_test_groups,
        cnn_confidence,
        list(GESTURE_LABEL_ORDER),
    )

    rf_all_test_probability = probability_for_windows_rf(rf_evaluation, split_windows["test"])
    cnn_all_test_probability = probability_for_windows_cnn(
        cnn_evaluation, split_windows["test"], cnn_scale
    )
    rf_end_to_end, rf_prediction_rows = end_to_end_trial_metrics(
        split_windows["test"],
        rf_all_test_probability,
        all_baselines,
        all_fallback,
        activation_on,
        rf_confidence,
        list(GESTURE_LABEL_ORDER),
        "Random Forest",
    )
    cnn_end_to_end, cnn_prediction_rows = end_to_end_trial_metrics(
        split_windows["test"],
        cnn_all_test_probability,
        all_baselines,
        all_fallback,
        activation_on,
        cnn_confidence,
        list(GESTURE_LABEL_ORDER),
        "1D CNN",
    )

    print("[5/5] 전체 데이터로 배포 모델을 만들고 결과를 저장합니다.", flush=True)
    all_active = all_gesture_windows(windows)
    _assert_active_labels(all_active, label_to_index, "deployment")
    x_rf_all, y_rf_all, _ = rf_arrays(all_active, label_to_index)
    rf_deployment = RandomForestClassifier(
        n_estimators=900,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_deployment.fit(x_rf_all, y_rf_all)

    all_classified = np.concatenate(
        [np.abs(np.asarray(window["classified"])).reshape(-1) for window in all_active]
    )
    deployment_scale = max(0.1, float(np.percentile(all_classified, 99.5)))
    x_cnn_all, y_cnn_all, _ = cnn_arrays(all_active, label_to_index, deployment_scale)
    cnn_deployment = train_cnn_deployment(
        x_cnn_all, y_cnn_all, len(GESTURE_LABEL_ORDER), best_epoch
    )

    recommended = (
        "Random Forest"
        if rf_end_to_end["trial_macro_f1"] >= cnn_end_to_end["trial_macro_f1"] - 0.02
        else "1D CNN"
    )
    metadata: dict[str, Any] = {
        "format_version": 3,
        "model_generation": "final_5class",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": "rest gate -> active index/middle/ring/fist classifier -> unknown",
        "recommended_model": recommended,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "window_stride_training": WINDOW_STRIDE,
        "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE_HZ,
        "filter": {
            "bandpass_hz": [FILTER_LOW_HZ, FINAL_FILTER_HIGH_HZ],
            "notch_hz": list(FINAL_NOTCH_FREQUENCIES_HZ),
            "notch_half_width_hz": NOTCH_HALF_WIDTH_HZ,
            "implementation": "per-window FFT mask",
        },
        "classification_preprocessing": {
            "normalization": "global_window_rms",
            "purpose": "reduce force/amplitude drift while retaining channel ratios",
        },
        "gesture_labels": list(GESTURE_LABEL_ORDER),
        "label_to_index": label_to_index,
        "rest_label": REST_LABEL,
        "activation": {
            "baseline_rms": all_fallback.tolist(),
            "on_ratio": activation_on,
            "off_ratio": activation_off,
            "live_baseline_calibration": True,
            "on_consecutive_windows": 2,
            "off_consecutive_windows": 5,
        },
        "rf": {
            "feature_version": RF_FEATURE_VERSION,
            "feature_names": feature_names(RF_FEATURE_VERSION),
            "confidence_threshold": rf_confidence,
        },
        "cnn": {
            "input_scale": deployment_scale,
            "confidence_threshold": cnn_confidence,
            "class_count": len(GESTURE_LABEL_ORDER),
            "deployment_epochs": int(max(8, best_epoch)),
        },
        "data": {
            "source": (
                "all dated 1 kHz RAW folders: valid manual records plus guided ACTION trials"
                if args.all_dates
                else "latest-date valid manual records plus guided ACTION trials"
            ),
            "include_all_dates": bool(args.all_dates),
            "manual_files_included": True,
            "sessions": preflight["completed_guided_sessions"],
            "trials": preflight["trial_counts"],
            "total_windows": len(windows),
            "active_deployment_windows": len(all_active),
            "split": split_info,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": str(torch.__version__),
        },
    }
    report: dict[str, Any] = {
        "created_at": metadata["created_at"],
        "preflight": preflight,
        "split": split_info,
        "split_trial_counts": {
            name: dict(
                Counter(
                    str(record["label"])
                    for record in records
                    if str(record["trial_id"]) in ids
                )
            )
            for name, ids in split.items()
        },
        "window_counts": {
            name: {
                "all": len(split_windows[name]),
                "active_gesture": len(active[name]),
            }
            for name in ("train", "validation", "test")
        },
        "activation": {
            "selected_on_ratio": activation_on,
            "selected_off_ratio": activation_off,
            "validation_selection": gate_selection,
            "validation": gate_metrics(
                split_windows["validation"], all_baselines, all_fallback, activation_on
            ),
            "test": gate_metrics(
                split_windows["test"], all_baselines, all_fallback, activation_on
            ),
        },
        "rf": {
            "active_gesture_test": rf_metrics,
            "end_to_end_four_state_test": rf_end_to_end,
        },
        "cnn": {
            "active_gesture_test": cnn_metrics,
            "end_to_end_four_state_test": cnn_end_to_end,
            "best_epoch": best_epoch,
            "history": history,
        },
        "recommended_model": recommended,
        "interpretation": {
            "primary_metric": "end-to-end four-state trial macro F1 on test split",
            "important_limitation": split_info["reliability"],
            "window_counts_are_not_independent_trials": True,
        },
    }

    pd.DataFrame(
        [
            {
                "trial_id": record["trial_id"],
                "session_id": record["session_id"],
                "label": record["label"],
                "split": next(
                    name for name, ids in split.items() if str(record["trial_id"]) in ids
                ),
            }
            for record in records
        ]
    ).to_csv(OUTPUT_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rf_prediction_rows + cnn_prediction_rows).to_csv(
        OUTPUT_DIR / "test_trial_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "cnn_training_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "feature": feature_names(RF_FEATURE_VERSION),
            "importance": rf_deployment.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        OUTPUT_DIR / "rf_feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    rf_temp = OUTPUT_DIR / "emg_raw_two_stage_rf.joblib.tmp"
    cnn_temp = OUTPUT_DIR / "emg_raw_two_stage_cnn.pt.tmp"
    joblib.dump({"model": rf_deployment, "metadata": metadata}, rf_temp, compress=3)
    torch.save(
        {"model_state_dict": cnn_deployment.state_dict(), "metadata": metadata},
        cnn_temp,
    )
    rf_temp.replace(OUTPUT_DIR / "emg_raw_two_stage_rf.joblib")
    cnn_temp.replace(OUTPUT_DIR / "emg_raw_two_stage_cnn.pt")
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_lines = [
        "퍼플팀 EMG 최종 5상태 모델 학습 결과",
        "",
        f"검증 방식: {split_info['mode']}",
        f"한계: {split_info['reliability']}",
        f"완료 세션: {preflight['completed_guided_sessions']}개",
        *[f"- {label}: {preflight['trial_counts'][label]}회" for label in ALL_LABELS],
        "",
        f"휴식 게이트 시험 균형정확도: {report['activation']['test']['balanced_accuracy']:.3f}",
        f"RF 5상태 시험 기록 정확도: {rf_end_to_end['trial_accuracy']:.3f}",
        f"RF 5상태 시험 macro F1: {rf_end_to_end['trial_macro_f1']:.3f}",
        f"CNN 5상태 시험 기록 정확도: {cnn_end_to_end['trial_accuracy']:.3f}",
        f"CNN 5상태 시험 macro F1: {cnn_end_to_end['trial_macro_f1']:.3f}",
        f"권장 모델: {recommended}",
        "",
        "정확도는 창(window) 개수가 아니라 독립된 동작 회차(trial) 기준으로 판독하세요.",
        "3세션 미만이면 새 전극 부착에 대한 성능은 확정할 수 없습니다.",
    ]
    (OUTPUT_DIR / "training_report.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8-sig"
    )
    print("\n".join(summary_lines))
    print(f"\nModels: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
