"""Shared preprocessing, RF features, and CNN architecture for RAW EMG."""

from __future__ import annotations

import numpy as np


SAMPLE_RATE_HZ = 1_000
WINDOW_SAMPLES = 1_000
WINDOW_STRIDE = 500
FILTER_LOW_HZ = 20.0
FILTER_HIGH_HZ = 250.0
NOTCH_FREQUENCIES_HZ = (60.0, 120.0, 180.0, 240.0)
NOTCH_HALF_WIDTH_HZ = 2.0
FINAL_FILTER_HIGH_HZ = 450.0
FINAL_NOTCH_FREQUENCIES_HZ = (60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0)
REST_LABEL = "휴식"
GESTURE_LABEL_ORDER = ("검지 움직임", "중지 움직임", "약지 움직임", "주먹 쥐기")


def spectral_mask(
    sample_count: int,
    *,
    low_hz: float = FILTER_LOW_HZ,
    high_hz: float = FILTER_HIGH_HZ,
    notch_frequencies_hz: tuple[float, ...] = NOTCH_FREQUENCIES_HZ,
    notch_half_width_hz: float = NOTCH_HALF_WIDTH_HZ,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / SAMPLE_RATE_HZ)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    for center in notch_frequencies_hz:
        mask &= np.abs(frequencies - center) > notch_half_width_hz
    return mask


def preprocess_window(
    window: np.ndarray,
    *,
    low_hz: float = FILTER_LOW_HZ,
    high_hz: float = FILTER_HIGH_HZ,
    notch_frequencies_hz: tuple[float, ...] = NOTCH_FREQUENCIES_HZ,
    notch_half_width_hz: float = NOTCH_HALF_WIDTH_HZ,
) -> np.ndarray:
    """DC removal plus identical FFT band-pass/notches for train and inference."""
    values = np.asarray(window, dtype=np.float32)
    if values.shape != (WINDOW_SAMPLES, 3):
        raise ValueError(f"Expected {(WINDOW_SAMPLES, 3)}, got {values.shape}")
    centered = values - np.mean(values, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    mask = spectral_mask(
        WINDOW_SAMPLES,
        low_hz=low_hz,
        high_hz=high_hz,
        notch_frequencies_hz=notch_frequencies_hz,
        notch_half_width_hz=notch_half_width_hz,
    )
    spectrum[~mask, :] = 0.0
    filtered = np.fft.irfft(spectrum, n=WINDOW_SAMPLES, axis=0)
    return filtered.astype(np.float32, copy=False)


def channel_rms(filtered_window: np.ndarray) -> np.ndarray:
    values = np.asarray(filtered_window, dtype=np.float64)
    return np.sqrt(np.mean(np.square(values), axis=0)).astype(np.float32)


def normalize_for_classifier(filtered_window: np.ndarray) -> np.ndarray:
    """Remove global contraction strength while preserving channel ratios."""
    values = np.asarray(filtered_window, dtype=np.float32)
    rms = channel_rms(values)
    global_rms = float(np.sqrt(np.mean(np.square(rms))))
    return (values / max(global_rms, 0.05)).astype(np.float32, copy=False)


def feature_names(feature_version: str = "legacy_v1") -> list[str]:
    per_channel = (
        "rms", "mav", "std", "median_abs", "p75_abs", "p90_abs",
        "p95_abs", "max_abs", "waveform_length", "zero_crossing_rate",
        "slope_sign_change_rate", "mean_frequency", "median_frequency",
        "power_20_55", "power_65_115", "power_125_250",
    )
    names = [
        f"a{channel}_{name}"
        for channel in range(3)
        for name in per_channel
    ]
    names.extend(
        [
            "total_rms", "rms_share_a0", "rms_share_a1", "rms_share_a2",
            "mav_share_a0", "mav_share_a1", "mav_share_a2",
            "log_rms_a0_over_a1", "log_rms_a0_over_a2",
            "log_rms_a1_over_a2", "corr_a0_a1", "corr_a0_a2",
            "corr_a1_a2",
        ]
    )
    if feature_version == "final_v2":
        names.extend(
            f"a{channel}_{name}"
            for channel in range(3)
            for name in ("power_250_450", "spectral_entropy")
        )
    return names


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def extract_rf_features(
    filtered_window: np.ndarray,
    feature_version: str = "legacy_v1",
) -> np.ndarray:
    values = np.asarray(filtered_window, dtype=np.float64)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / SAMPLE_RATE_HZ)
    result: list[float] = []
    rms_values: list[float] = []
    mav_values: list[float] = []

    for channel_index in range(3):
        channel = values[:, channel_index]
        absolute = np.abs(channel)
        rms = float(np.sqrt(np.mean(np.square(channel))))
        mav = float(np.mean(absolute))
        rms_values.append(rms)
        mav_values.append(mav)

        difference = np.diff(channel)
        crossing_threshold = max(0.5, 0.02 * rms)
        zero_crossings = np.mean(
            ((channel[:-1] >= 0) != (channel[1:] >= 0))
            & (np.abs(difference) >= crossing_threshold)
        )
        if len(difference) >= 2:
            slope_changes = np.mean(
                ((difference[:-1] >= 0) != (difference[1:] >= 0))
                & (np.abs(difference[:-1] - difference[1:]) >= crossing_threshold)
            )
        else:
            slope_changes = 0.0

        spectrum = np.fft.rfft(channel)
        power = np.square(np.abs(spectrum))
        high_hz = FINAL_FILTER_HIGH_HZ if feature_version == "final_v2" else FILTER_HIGH_HZ
        power[(frequencies < FILTER_LOW_HZ) | (frequencies > high_hz)] = 0.0
        total_power = float(np.sum(power)) + 1e-12
        mean_frequency = float(np.sum(frequencies * power) / total_power)
        cumulative = np.cumsum(power)
        median_frequency = float(
            frequencies[min(int(np.searchsorted(cumulative, cumulative[-1] * 0.5)), len(frequencies) - 1)]
        )

        def band_share(low: float, high: float) -> float:
            selected = (frequencies >= low) & (frequencies <= high)
            return float(np.sum(power[selected]) / total_power)

        result.extend(
            [
                rms,
                mav,
                float(np.std(channel)),
                float(np.median(absolute)),
                float(np.percentile(absolute, 75)),
                float(np.percentile(absolute, 90)),
                float(np.percentile(absolute, 95)),
                float(np.max(absolute)),
                float(np.mean(np.abs(difference))) if len(difference) else 0.0,
                float(zero_crossings),
                float(slope_changes),
                mean_frequency,
                median_frequency,
                band_share(20.0, 55.0),
                band_share(65.0, 115.0),
                band_share(125.0, 250.0),
            ]
        )

    rms_array = np.asarray(rms_values, dtype=np.float64)
    mav_array = np.asarray(mav_values, dtype=np.float64)
    rms_share = rms_array / (np.sum(rms_array) + 1e-12)
    mav_share = mav_array / (np.sum(mav_array) + 1e-12)
    result.extend(
        [
            float(np.sum(rms_array)),
            *[float(value) for value in rms_share],
            *[float(value) for value in mav_share],
            float(np.log((rms_array[0] + 1e-6) / (rms_array[1] + 1e-6))),
            float(np.log((rms_array[0] + 1e-6) / (rms_array[2] + 1e-6))),
            float(np.log((rms_array[1] + 1e-6) / (rms_array[2] + 1e-6))),
            _safe_correlation(values[:, 0], values[:, 1]),
            _safe_correlation(values[:, 0], values[:, 2]),
            _safe_correlation(values[:, 1], values[:, 2]),
        ]
    )
    if feature_version == "final_v2":
        for channel_index in range(3):
            channel = values[:, channel_index]
            power = np.square(np.abs(np.fft.rfft(channel)))
            selected_total = (frequencies >= FILTER_LOW_HZ) & (
                frequencies <= FINAL_FILTER_HIGH_HZ
            )
            selected_high = (frequencies >= 250.0) & (
                frequencies <= FINAL_FILTER_HIGH_HZ
            )
            total_power = float(np.sum(power[selected_total])) + 1e-12
            normalized_power = power[selected_total] / total_power
            positive = normalized_power[normalized_power > 0]
            entropy = -float(np.sum(positive * np.log(positive)))
            entropy /= max(float(np.log(max(len(normalized_power), 2))), 1e-12)
            result.extend(
                [
                    float(np.sum(power[selected_high]) / total_power),
                    entropy,
                ]
            )
    return np.nan_to_num(np.asarray(result, dtype=np.float32))
