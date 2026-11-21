"""Screen historical inference inputs for quality and distribution changes."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = (
    BASE_DIR / "data" / "inference" / "history_through_2026-11-15.csv"
)

METRICS = [
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "rain_mm",
    "wind_speed_ms",
    "us_aqi",
    "pm2_5_ug_m3",
    "pm10_ug_m3",
    "uv_index",
]

REFERENCE_START = pd.Timestamp("2025-11-01", tz="UTC")
REFERENCE_END = pd.Timestamp("2026-08-01", tz="UTC")
WINDOW_HOURS = 168

# Initial screening rules, not statistically validated retraining thresholds.
MEAN_SHIFT_THRESHOLD = 1.0
OUTSIDE_RANGE_THRESHOLD = 0.05


def read_history(path):
    frame = pd.read_csv(path)

    required = ["hour_utc", *METRICS]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame["hour_utc"] = pd.to_datetime(
        frame["hour_utc"],
        utc=True,
        errors="raise",
    )

    if frame.empty or frame["hour_utc"].isna().any():
        raise ValueError("History is empty or contains invalid timestamps.")

    if frame["hour_utc"].duplicated().any():
        raise ValueError("History contains duplicate hours.")

    if not frame["hour_utc"].eq(
        frame["hour_utc"].dt.floor("h")
    ).all():
        raise ValueError("History timestamps must fall on exact hours.")

    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")

    return frame.set_index("hour_utc").sort_index()


def assess_drift(history_path):
    """Return a JSON-compatible monitoring report without writing files."""
    frame = read_history(Path(history_path))

    reference = frame.loc[
        (frame.index >= REFERENCE_START)
        & (frame.index < REFERENCE_END),
        METRICS,
    ]

    expected_reference = pd.date_range(
        REFERENCE_START,
        REFERENCE_END,
        inclusive="left",
        freq="h",
    )

    if not reference.index.equals(expected_reference):
        raise ValueError("Training reference has missing or unexpected hours.")

    if not np.isfinite(reference.to_numpy(dtype=float)).all():
        raise ValueError("Training reference contains invalid metric values.")

    latest = frame.index.max()
    expected_window = pd.date_range(
        end=latest,
        periods=WINDOW_HOURS,
        freq="h",
    )

    if expected_window.min() < REFERENCE_END:
        raise ValueError("Monitoring window overlaps the training reference.")

    missing_hours = expected_window.difference(frame.index)
    recent = frame.reindex(expected_window)[METRICS]

    now = pd.Timestamp.now(tz="UTC")
    age_hours = (now - latest).total_seconds() / 3600

    if age_hours < 0:
        raise ValueError("Latest input timestamp is in the future.")

    quality_counts = {}
    metric_results = []

    for metric in METRICS:
        old = reference[metric].to_numpy(dtype=float)
        new = recent[metric].to_numpy(dtype=float)

        valid = np.isfinite(new)
        invalid_count = int((~valid).sum())
        quality_counts[metric] = invalid_count

        old_mean = float(old.mean())
        old_std = float(old.std(ddof=0))
        old_min = float(old.min())
        old_max = float(old.max())

        result = {
            "metric": metric,
            "reference_mean": old_mean,
            "reference_std": old_std,
            "reference_min": old_min,
            "reference_max": old_max,
            "recent_mean": None,
            "absolute_standardized_mean_shift": None,
            "outside_training_range_fraction": None,
            "invalid_recent_values": invalid_count,
            "screening_flag": None,
            "reasons": [],
        }

        # Incomplete windows are quality failures, not valid drift estimates.
        if invalid_count:
            result["reasons"].append(
                "Distribution screening skipped: incomplete or invalid window."
            )
            metric_results.append(result)
            continue

        recent_mean = float(new.mean())
        outside_fraction = float(
            ((new < old_min) | (new > old_max)).mean()
        )

        result["recent_mean"] = recent_mean
        result["outside_training_range_fraction"] = outside_fraction

        reasons = []

        if old_std > 0:
            shift = abs(recent_mean - old_mean) / old_std
            result["absolute_standardized_mean_shift"] = float(shift)

            if shift >= MEAN_SHIFT_THRESHOLD:
                reasons.append(
                    "Recent mean differs by at least one training "
                    "standard deviation."
                )
        elif not np.equal(new, old_mean).all():
            reasons.append(
                "Training reference was constant; recent values differ."
            )

        if outside_fraction >= OUTSIDE_RANGE_THRESHOLD:
            reasons.append(
                "At least 5% of recent values are outside the training range."
            )

        result["screening_flag"] = bool(reasons)
        result["reasons"] = reasons
        metric_results.append(result)

    quality_passed = (
        len(missing_hours) == 0
        and all(count == 0 for count in quality_counts.values())
    )
    flagged = [
        result["metric"]
        for result in metric_results
        if result["screening_flag"] is True
    ]

    if not quality_passed:
        status = "input_quality_failed"
    elif flagged:
        status = "review_distribution_changes"
    else:
        status = "no_screening_flags"

    return {
        "policy_version": 1,
        "generated_at_utc": now.isoformat(),
        "mode": "retrospective_historical_demo",
        "history_path": str(Path(history_path).resolve()),
        "status": status,
        "reference_start_utc": REFERENCE_START.isoformat(),
        "reference_end_exclusive_utc": REFERENCE_END.isoformat(),
        "reference_rows": len(reference),
        "window_start_utc": expected_window.min().isoformat(),
        "window_end_utc": latest.isoformat(),
        "window_hours": WINDOW_HOURS,
        "input_age_hours": float(age_hours),
        "quality_passed": quality_passed,
        "missing_hours": [value.isoformat() for value in missing_hours],
        "invalid_values_by_metric": quality_counts,
        "flagged_metrics": flagged,
        "metric_results": metric_results,
        "screening_rules": {
            "absolute_standardized_mean_shift_gte": MEAN_SHIFT_THRESHOLD,
            "outside_training_range_fraction_gte": OUTSIDE_RANGE_THRESHOLD,
            "threshold_status": "initial_heuristics_not_validated",
        },
        "performance_drift_assessed": False,
        "retraining_triggered": False,
        "limitations": [
            "Historical input screening, not live monitoring.",
            "Seasonality can explain differences from the full training period.",
            "Screening flags do not establish reduced forecast accuracy.",
            "No forecast outcomes were compared with predictions.",
            "Wind direction is excluded because it requires circular analysis.",
            "Source metadata and physical bounds are not checked here.",
            "Absence of flags does not prove absence of drift.",
        ],
    }


def main():
    print("AgentWeather — Historical Input Monitoring")

    try:
        report = assess_drift(HISTORY_PATH)

        print(f"\nTraining reference rows: {report['reference_rows']}")
        print(f"Monitoring window: {report['window_hours']} hours")
        print(f"First monitored hour: {report['window_start_utc']}")
        print(f"Last monitored hour: {report['window_end_utc']}")
        print(f"Input age: {report['input_age_hours']:.1f} hours")
        print(f"Missing hours: {len(report['missing_hours'])}")
        print(f"Input quality passed: {report['quality_passed']}")

        columns = [
            "metric",
            "reference_mean",
            "recent_mean",
            "absolute_standardized_mean_shift",
            "outside_training_range_fraction",
            "screening_flag",
        ]

        table = pd.DataFrame(report["metric_results"])
        print("\nDistribution screening:")
        print(table[columns].round(4).to_string(index=False))

        print(f"\nStatus: {report['status']}")
        print(
            "Flagged metrics: "
            + (", ".join(report["flagged_metrics"]) or "None")
        )

        output_dir = BASE_DIR / "data" / "monitoring"
        output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        output = output_dir / (
            f"historical_drift_{stamp}_{uuid4().hex[:8]}.json"
        )

        with output.open("x", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")

        print(f"\nSaved to: {output}")
        print("Seasonal differences may produce screening flags.")
        print("Forecast-error drift was not assessed.")
        print("No retraining, model changes or Hopsworks upload.")

        return 0 if report["quality_passed"] else 1

    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Monitoring failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())