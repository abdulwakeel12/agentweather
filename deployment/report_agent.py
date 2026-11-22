"""Create a factual report from a historical forecast and its alerts."""

import argparse
import csv
import json
import math
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
PKT = timezone(timedelta(hours=5), name="PKT")

METRICS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
    "rain_probability",
]


def parse_time(value):
    parsed = datetime.fromisoformat(str(value))

    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a timezone.")

    return parsed.astimezone(UTC)


def display_pkt(value):
    return value.astimezone(PKT).strftime("%d %b %Y, %H:%M %Z")


def read_forecast(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 24:
        raise ValueError("Expected exactly 24 forecast rows.")

    rows.sort(key=lambda row: int(row["horizon_hours"]))

    horizons = [int(row["horizon_hours"]) for row in rows]
    if horizons != list(range(1, 25)):
        raise ValueError("Forecast horizons must be exactly 1–24.")

    origins = {
        parse_time(row["prediction_time_utc"])
        for row in rows
    }

    if len(origins) != 1:
        raise ValueError("Expected one prediction origin.")

    origin = origins.pop()

    for horizon, row in enumerate(rows, start=1):
        if row["forecast_mode"] != "retrospective_historical_demo":
            raise ValueError(
                "This report supports historical demonstrations only."
            )

        valid_time = parse_time(row["valid_time_utc"])

        if valid_time != origin + timedelta(hours=horizon):
            raise ValueError("Forecast target timestamps are incorrect.")

        for metric in METRICS:
            value = float(row[metric])

            if not math.isfinite(value):
                raise ValueError(f"Invalid forecast value: {metric}")

            row[metric] = value

        for metric in (
            "us_aqi",
            "precipitation_mm",
            "uv_index",
            "wind_speed_ms",
        ):
            if row[metric] < 0:
                raise ValueError(f"Negative forecast value: {metric}")

        if not 0 <= row["humidity_pct"] <= 100:
            raise ValueError("Humidity must be between 0 and 100.")

        if not 0 <= row["rain_probability"] <= 1:
            raise ValueError("Rain probabilities must be between 0 and 1.")

    run_ids = {row.get("agent_run_id", "") for row in rows}

    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise ValueError("Expected one non-empty agent run ID.")

    return rows, origin, run_ids.pop()


def generate_report(forecast_path, alert_path):
    """Return report text and metadata without writing files."""
    forecast_path = Path(forecast_path)
    alert_path = Path(alert_path)

    rows, origin, run_id = read_forecast(forecast_path)

    with alert_path.open(encoding="utf-8") as handle:
        alert_report = json.load(handle)

    if not isinstance(alert_report, dict):
        raise TypeError("Expected an alert report object.")

    if alert_report.get("agent_run_id") != run_id:
        raise ValueError(
            "Forecast and alerts belong to different workflow runs."
        )

    alert_origin = parse_time(alert_report["prediction_time_utc"])

    if alert_origin != origin:
        raise ValueError("Forecast and alert prediction origins differ.")

    if alert_report.get("mode") != "retrospective_historical_demo":
        raise ValueError(
            "Alert report is not marked as a historical demonstration."
        )

    alerts = alert_report["alerts"]

    if not isinstance(alerts, list):
        raise TypeError("Expected an alert list.")

    def limits(metric):
        values = [row[metric] for row in rows]
        return min(values), max(values)

    temp_low, temp_high = limits("temperature_c")
    humidity_low, humidity_high = limits("humidity_pct")
    wind_low, wind_high = limits("wind_speed_ms")
    aqi_low, aqi_high = limits("us_aqi")
    rain_low, rain_high = limits("rain_probability")

    precipitation_total = sum(
        row["precipitation_mm"] for row in rows
    )

    peak_uv_row = max(rows, key=lambda row: row["uv_index"])
    peak_uv_time = parse_time(peak_uv_row["valid_time_utc"])

    first_target = parse_time(rows[0]["valid_time_utc"])
    last_target = parse_time(rows[-1]["valid_time_utc"])
    generated_at = datetime.now(UTC)

    lines = [
        "# AgentWeather — Karachi Weather Report",
        "",
        "**Historical demonstration — not a current forecast or warning.**",
        "",
        f"Prediction origin: {display_pkt(origin)}",
        (
            f"Forecast period: {display_pkt(first_target)} "
            f"to {display_pkt(last_target)}"
        ),
        f"Report generated: {generated_at.isoformat()}",
        f"Workflow run: `{run_id}`",
        "",
        "## Weather outlook",
        "",
        (
            "For this historical forecast period, the models predict "
            f"temperatures between {temp_low:.1f} and {temp_high:.1f} °C. "
            f"Relative humidity ranges from {humidity_low:.0f}% to "
            f"{humidity_high:.0f}%, while wind speeds range from "
            f"{wind_low:.1f} to {wind_high:.1f} m/s."
        ),
        "",
        (
            "The sum of the 24 hourly precipitation predictions is "
            f"{precipitation_total:.2f} mm. Estimated rain probabilities "
            f"range from {rain_low:.1%} to {rain_high:.1%} for individual "
            "target hours. These are not the probability of rain at any "
            "time during the entire day. Probability calibration remains "
            "unestablished, and rain alerts are disabled."
        ),
        "",
        "## Air quality and UV",
        "",
        (
            f"Predicted modelled US AQI ranges from {aqi_low:.1f} to "
            f"{aqi_high:.1f}. The highest predicted UV Index is "
            f"{peak_uv_row['uv_index']:.2f}, at "
            f"{display_pkt(peak_uv_time)}."
        ),
        "",
        "## Historical alert evaluation",
        "",
    ]

    if alerts:
        for alert in alerts:
            if not isinstance(alert, dict):
                raise TypeError("Each alert must be an object.")

            metric = alert["metric"]

            if metric not in ("us_aqi", "uv_index"):
                raise ValueError(f"Unsupported alert metric: {metric}")

            first = parse_time(alert["first_target_time_utc"])
            last = parse_time(alert["last_target_time_utc"])

            if not first_target <= first <= last <= last_target:
                raise ValueError(
                    "Alert window is outside the forecast period."
                )

            peak = float(alert["peak_predicted_value"])

            if not math.isfinite(peak):
                raise ValueError("Invalid alert peak.")

            label = "US AQI" if metric == "us_aqi" else "UV Index"

            lines.append(
                f"- **{label}: {alert['category_at_peak']} "
                "at the window peak.** "
                f"{alert['flagged_hours']} flagged hours, from "
                f"{display_pkt(first)} to {display_pkt(last)}. "
                f"Peak predicted value: {peak:.2f}."
            )
    else:
        lines.append(
            "No windows met the enabled AQI and UV rules. "
            "This does not establish that all conditions are safe."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- This report was generated retrospectively using "
                "historical inputs; it was not issued at the "
                "prediction origin."
            ),
            (
                "- Values are model predictions, not observed conditions. "
                "Historical weather inputs are reanalysis data, and "
                "air-quality inputs are modelled estimates."
            ),
            (
                "- Forecast accuracy varies by metric and horizon. "
                "The selected models did not consistently outperform "
                "simple baselines on the October test period."
            ),
            (
                "- Wind direction is omitted from this narrative until "
                "its display policy for calm or ambiguous predictions "
                "is defined."
            ),
            "- No notifications were sent. No LLM was used.",
            "",
        ]
    )

    return {
        "agent_run_id": run_id,
        "prediction_time_utc": origin.isoformat(),
        "generated_at_utc": generated_at.isoformat(),
        "mode": "retrospective_historical_demo",
        "report_method": "deterministic_template",
        "forecast_rows": len(rows),
        "alert_windows": len(alerts),
        "text": "\n".join(lines),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a historical AgentWeather report."
    )
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--alerts", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = generate_report(args.forecast, args.alerts)

        output_dir = BASE_DIR / "data" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        output = output_dir / (
            f"historical_report_{stamp}_{uuid4().hex[:8]}.md"
        )

        # Exclusive creation preserves any existing report.
        with output.open("x", encoding="utf-8") as handle:
            handle.write(report["text"])

        print(report["text"])
        print(f"Saved to: {output}")
        print("No model changes, LLM calls or Hopsworks upload.")
        return 0

    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Report generation failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())