import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

AQI_SOURCE = "https://www.airnow.gov/aqi/aqi-basics/"
UV_SOURCE = "https://www.epa.gov/sunsafety/uv-index-scale-0"


def aqi_category(value):
    # Project policy: apply continuous boundaries without rounding predictions.
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for Sensitive Groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very Unhealthy"
    if value <= 500:
        return "Hazardous"
    return "Beyond the published AQI scale"


def uv_category(value):
    if value < 3:
        return "Low"
    if value < 6:
        return "Moderate"
    if value < 8:
        return "High"
    if value < 11:
        return "Very High"
    return "Extreme"


def evaluate_alerts(forecast):
    """Evaluate historical-demo forecasts without sending notifications."""
    frame = forecast.copy()

    required = {
        "prediction_time_utc",
        "valid_time_utc",
        "horizon_hours",
        "forecast_mode",
        "us_aqi",
        "uv_index",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    if not frame["forecast_mode"].eq("retrospective_historical_demo").all():
        raise ValueError("Only explicitly labelled historical demos are supported.")

    for column in ("prediction_time_utc", "valid_time_utc"):
        frame[column] = pd.to_datetime(
            frame[column], utc=True, errors="raise"
        )
        if frame[column].isna().any():
            raise ValueError(f"Missing timestamps: {column}")

    frame["horizon_hours"] = pd.to_numeric(
        frame["horizon_hours"], errors="raise"
    )
    frame = frame.sort_values("horizon_hours")

    if frame["horizon_hours"].tolist() != list(range(1, 25)):
        raise ValueError("Expected exactly one row for each horizon 1–24.")

    if frame["prediction_time_utc"].nunique() != 1:
        raise ValueError("Expected exactly one prediction origin.")

    origin = frame["prediction_time_utc"].iloc[0]
    expected = pd.date_range(
        origin + pd.Timedelta(hours=1), periods=24, freq="h"
    )
    if not pd.DatetimeIndex(frame["valid_time_utc"]).equals(expected):
        raise ValueError("Forecast timestamps do not match their horizons.")

    for column in ("us_aqi", "uv_index"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid values in {column}")

    rules = [
        ("us_aqi", lambda value: value > 100, aqi_category, AQI_SOURCE),
        ("uv_index", lambda value: value >= 3, uv_category, UV_SOURCE),
    ]
    alerts = []

    # Combine consecutive flagged hours into one alert window.
    for metric, triggered, category, source in rules:
        groups = []
        current = []

        for _, row in frame.iterrows():
            if triggered(float(row[metric])):
                current.append(row)
            elif current:
                groups.append(current)
                current = []

        if current:
            groups.append(current)

        for group in groups:
            peak_row = max(group, key=lambda row: float(row[metric]))
            peak = float(peak_row[metric])
            first = group[0]["valid_time_utc"]
            last = group[-1]["valid_time_utc"]

            alerts.append({
                "metric": metric,
                "status": "historical_demo_only",
                "category_at_peak": category(peak),
                "first_target_time_utc": first.isoformat(),
                "last_target_time_utc": last.isoformat(),
                "first_target_time_pkt": first.tz_convert(
                    "Asia/Karachi"
                ).isoformat(),
                "last_target_time_pkt": last.tz_convert(
                    "Asia/Karachi"
                ).isoformat(),
                "flagged_hours": len(group),
                "peak_predicted_value": peak,
                "peak_target_time_utc": peak_row["valid_time_utc"].isoformat(),
                "category_reference": source,
                "message": (
                    f"Historical model predictions crossed the {metric} "
                    "demo threshold. This is not a current warning."
                ),
            })

    return {
        "policy_version": 1,
        "mode": "retrospective_historical_demo",
        "prediction_time_utc": origin.isoformat(),
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "forecast_rows_checked": len(frame),
        "rules": {
            "us_aqi": "Unrounded predicted value > 100",
            "uv_index": "Unrounded predicted value >= 3",
        },
        "alerts": alerts,
        "disabled_rules": {
            "rain": "Probability calibration and operational threshold unresolved.",
            "temperature": "Local alert policy not yet established.",
            "humidity": "Alert policy not yet established.",
            "wind": "Local alert policy and calm-wind handling not yet established.",
        },
        "limitations": [
            "Thresholds are applied to model estimates, not measurements.",
            "These forecasts are historical, not current.",
            "No alert accuracy assessment has been completed.",
            "No flagged hours does not establish that conditions were safe.",
        ],
        "notifications_sent": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate historical forecast alert rules."
    )
    parser.add_argument("--forecast", type=Path, required=True)
    args = parser.parse_args()

    forecast = pd.read_csv(args.forecast)
    report = evaluate_alerts(forecast)
    report["forecast_file"] = str(args.forecast.resolve())

    output_dir = BASE_DIR / "data/alerts"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        + "_" + uuid4().hex[:8]
    )
    output = output_dir / f"historical_alerts_{run_id}.json"
    temporary = output.with_suffix(".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    temporary.replace(output)

    print("\nAgentWeather — Historical Alert Evaluation")
    print("Forecast rows checked:", report["forecast_rows_checked"])
    print("Alert windows:", len(report["alerts"]))

    for alert in report["alerts"]:
        print(
            f"\n{alert['metric']}: {alert['category_at_peak']}"
            f"\n  First target: {alert['first_target_time_pkt']}"
            f"\n  Last target:  {alert['last_target_time_pkt']}"
            f"\n  Flagged hours: {alert['flagged_hours']}"
            f"\n  Peak predicted value: {alert['peak_predicted_value']:.2f}"
        )

    print(f"\nSaved to: {output}")
    print("Historical demo only. No current warnings issued.")
    print("Rain alerts disabled. No notifications or API calls.")


if __name__ == "__main__":
    main()