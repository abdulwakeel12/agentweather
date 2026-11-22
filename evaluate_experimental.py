import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_RUN = (
    BASE_DIR
    / "data/experimental_forecasts/20261122T111628036238"
)

CONTINUOUS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
]
METRICS = [*CONTINUOUS, "rain_probability", "wind_direction_deg"]

WEATHER = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precipitation_mm",
    "rain": "rain_mm",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
}

AIR = {
    "us_aqi": "us_aqi",
    "uv_index": "uv_index",
}

UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "precipitation": "mm",
    "rain": "mm",
    "wind_speed_10m": "m/s",
    "wind_direction_10m": "°",
    "us_aqi": "USAQI",
    "uv_index": "",
}


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def read_predictions(path, generated_column):
    frame = pd.read_csv(path)

    for column in (
        "prediction_time_utc",
        "valid_time_utc",
        generated_column,
    ):
        frame[column] = pd.to_datetime(
            frame[column], utc=True, errors="raise"
        )
        if frame[column].isna().any():
            raise ValueError(f"Missing timestamps in {path.name}: {column}")

    if not frame["forecast_mode"].eq("experimental_mixed_source").all():
        raise ValueError(f"Unexpected forecast mode: {path.name}")

    frame[METRICS] = frame[METRICS].apply(
        pd.to_numeric, errors="raise"
    )

    required_finite = [*CONTINUOUS, "rain_probability"]
    if not np.isfinite(
        frame[required_finite].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"Invalid predictions: {path.name}")

    if np.isinf(frame["wind_direction_deg"].to_numpy(float)).any():
        raise ValueError("Infinite wind directions.")

    if not frame["rain_probability"].between(0, 1).all():
        raise ValueError("Rain probabilities must be between zero and one.")

    return frame


def validate_horizons(frame, origin, generated, generated_column):
    frame = frame.sort_values("horizon_hours")

    if frame["horizon_hours"].tolist() != list(range(1, 25)):
        raise ValueError("Expected exactly one row per horizon 1–24.")

    if not frame["prediction_time_utc"].eq(origin).all():
        raise ValueError("Input cutoffs differ.")

    if not frame[generated_column].eq(generated).all():
        raise ValueError("Generation timestamps differ.")

    expected = origin + pd.to_timedelta(
        frame["horizon_hours"].to_numpy(), unit="h"
    )
    if not pd.DatetimeIndex(frame["valid_time_utc"]).equals(
        pd.DatetimeIndex(expected)
    ):
        raise ValueError("Incorrect target timestamps.")

    return frame.set_index("valid_time_utc")


def fetch_reference(name, endpoint, variables, extra, times, output):
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "timezone": "UTC",
        "start_date": times.min().strftime("%Y-%m-%d"),
        "end_date": times.max().strftime("%Y-%m-%d"),
        "hourly": ",".join(variables),
        **extra,
    }

    print(f"Fetching {name} references...", flush=True)
    response = requests.get(
        endpoint, params=params, timeout=(15, 120)
    )

    (output / f"{name}_response.json").write_text(
        response.text, encoding="utf-8"
    )
    save_json(
        output / f"{name}_request.json",
        {
            "endpoint": endpoint,
            "parameters": params,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "http_status": response.status_code,
        },
    )

    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise ValueError(payload.get("reason", f"{name} API error."))

    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Expected UTC reference timestamps.")

    for variable in variables:
        if payload["hourly_units"].get(variable) != UNITS[variable]:
            raise ValueError(f"Unexpected reference unit: {variable}")

    frame = pd.DataFrame(payload["hourly"])
    timestamps = pd.to_datetime(
        frame.pop("time"), utc=True, errors="raise"
    )

    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Invalid reference timestamps.")

    if not timestamps.eq(timestamps.dt.floor("h")).all():
        raise ValueError("Reference timestamps are not hourly.")

    frame.index = pd.DatetimeIndex(timestamps, name="valid_time_utc")
    frame = frame[list(variables)].rename(columns=variables)
    frame = frame.apply(pd.to_numeric, errors="raise")

    if np.isinf(frame.to_numpy(float)).any():
        raise ValueError("Infinite reference values.")

    # Never score API values for target times outside the requested set.
    return frame.reindex(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_dir = args.run.resolve()

    forecast_path = run_dir / "forecast.csv"
    baseline_path = run_dir / "baseline_predictions.csv"

    hashes = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (forecast_path, baseline_path)
    }

    forecast = read_predictions(forecast_path, "generated_at_utc")
    baselines = read_predictions(
        baseline_path, "forecast_generated_at_utc"
    )

    if forecast.empty:
        raise ValueError("Forecast is empty.")

    origin = forecast["prediction_time_utc"].iloc[0]
    generated = forecast["generated_at_utc"].iloc[0]
    checked = pd.Timestamp.now(tz="UTC")

    if generated > checked or origin > generated:
        raise ValueError("Invalid forecast generation or cutoff time.")

    predictions = {
        "selected_models": validate_horizons(
            forecast, origin, generated, "generated_at_utc"
        )
    }

    if set(baselines["baseline"]) != {"persistence", "previous_day"}:
        raise ValueError("Expected persistence and previous-day baselines.")

    for method in ("persistence", "previous_day"):
        predictions[method] = validate_horizons(
            baselines.loc[baselines["baseline"].eq(method)],
            origin,
            generated,
            "forecast_generated_at_utc",
        )

    targets = predictions["selected_models"].index
    interval_starts = targets - pd.Timedelta(hours=1)

    # Conservative common timing rule for all outputs.
    prospective = interval_starts >= generated
    elapsed = targets <= checked
    ready_times = targets[prospective & elapsed]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output = run_dir / "evaluations" / stamp
    output.mkdir(parents=True, exist_ok=False)

    timing = pd.DataFrame(
        {
            "valid_time_utc": targets,
            "preceding_hour_start_utc": interval_starts,
            "prospective_interval": prospective,
            "target_elapsed": elapsed,
        }
    )
    timing.to_csv(output / "timing.csv", index=False)

    save_json(
        output / "evaluation_metadata.json",
        {
            "checked_at_utc": checked.isoformat(),
            "forecast_generated_at_utc": generated.isoformat(),
            "input_cutoff_utc": origin.isoformat(),
            "input_hashes": hashes,
            "timing_rule": (
                "Target minus one hour must be at or after generation; "
                "target must also have elapsed."
            ),
            "reference_weather": "Open-Meteo ERA5 reanalysis",
            "reference_air": "Open-Meteo CAMS Global model estimates",
            "station_observed_accuracy": False,
        },
    )

    print("AgentWeather — Experimental Reference Evaluation")
    print("Forecast generated:", generated)
    print("Checked at:", checked)
    print("Excluded by timing rule:", int((~prospective).sum()))
    print("Prospective targets still in future:", int((prospective & ~elapsed).sum()))
    print("Elapsed eligible targets:", len(ready_times))

    if ready_times.empty:
        print("\nPENDING: no eligible target intervals have ended.")
        print("No API requests were needed.")
        print("Saved timing report:", output)
        return

    weather = fetch_reference(
        "era5",
        "https://archive-api.open-meteo.com/v1/archive",
        WEATHER,
        {
            "models": "era5",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        },
        ready_times,
        output,
    )

    air = fetch_reference(
        "cams",
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        AIR,
        {"domains": "cams_global"},
        ready_times,
        output,
    )

    reference = weather.join(air)
    reference.to_csv(output / "reference_values.csv")

    # Preserve missing rainfall; missing is not a dry event.
    reference["rain_probability"] = np.where(
        reference["rain_mm"].notna(),
        reference["rain_mm"].ge(0.1).astype(float),
        np.nan,
    )

    score_rows = []
    detail_rows = []

    for metric in METRICS:
        actual = reference[metric].to_numpy(float)
        predicted = {
            method: frame.loc[ready_times, metric].to_numpy(float)
            for method, frame in predictions.items()
        }

        reference_available = np.isfinite(actual)
        usable = reference_available.copy()

        for values in predicted.values():
            usable &= np.isfinite(values)

        if metric == "wind_direction_deg":
            speed = reference["wind_speed_ms"].to_numpy(float)
            usable &= np.isfinite(speed) & (speed > 0)

            # Use a common non-calm subset for all compared methods.
            for frame in predictions.values():
                usable &= (
                    frame.loc[ready_times, "wind_speed_ms"].to_numpy(float) > 0
                )

        for method, values in predicted.items():
            error = values - actual

            if metric == "wind_direction_deg":
                error = (error + 180) % 360 - 180

            selected = error[usable]
            count = len(selected)

            row = {
                "metric": metric,
                "method": method,
                "eligible_elapsed_hours": len(ready_times),
                "reference_pending_hours": int((~reference_available).sum()),
                "excluded_other_hours": int((reference_available & ~usable).sum()),
                "scored_hours": count,
                "mae": np.nan,
                "rmse": np.nan,
                "brier_score": np.nan,
            }

            if count:
                if metric == "rain_probability":
                    row["brier_score"] = float(np.mean(selected**2))
                else:
                    row["mae"] = float(np.abs(selected).mean())
                    row["rmse"] = float(np.sqrt(np.mean(selected**2)))

            score_rows.append(row)

            for index, target in enumerate(ready_times):
                detail_rows.append(
                    {
                        "metric": metric,
                        "method": method,
                        "valid_time_utc": target,
                        "reference": actual[index],
                        "prediction": values[index],
                        "scored": bool(usable[index]),
                        "error": error[index] if usable[index] else np.nan,
                    }
                )

    scores = pd.DataFrame(score_rows)
    scores.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(
        output / "comparison_by_target.csv", index=False
    )

    print("\nReference comparison:")
    print(scores.round(4).to_string(index=False))
    print("\nSaved to:", output)
    print("Missing references remain pending.")
    print("Scores compare with reanalysis/model estimates, not station truth.")
    print("Wind errors are angular degrees; rain uses Brier score.")
    print("These scores summarize one run, not established reliability.")
    print("No models fitted, predictions changed or Hopsworks uploads.")


if __name__ == "__main__":
    main()