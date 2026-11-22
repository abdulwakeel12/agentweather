import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent

SNAPSHOT_PATH = (
    BASE_DIR
    / "data/recent_inputs/20261122T104510468620_e198e792"
    / "recent_hourly_inputs.csv"
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

VARIABLES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precipitation_mm",
    "rain": "rain_mm",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
}

EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "precipitation": "mm",
    "rain": "mm",
    "wind_speed_10m": "m/s",
    "wind_direction_10m": "°",
}

METRICS = list(VARIABLES.values())


def save_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def set_time_index(frame, time_column):
    frame = frame.copy()
    times = pd.to_datetime(
        frame.pop(time_column),
        utc=True,
        errors="raise",
    )

    if times.isna().any() or times.duplicated().any():
        raise ValueError("Missing or duplicated timestamps.")

    if not times.eq(times.dt.floor("h")).all():
        raise ValueError("Timestamps must be aligned to hourly boundaries.")

    frame.index = pd.DatetimeIndex(times, name="hour_utc")
    return frame.sort_index()


def load_snapshot():
    frame = pd.read_csv(SNAPSHOT_PATH)

    required = {
        "hour_utc",
        "weather_source",
        "weather_grid_latitude",
        "weather_grid_longitude",
        *METRICS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Snapshot is missing columns: {sorted(missing)}")

    frame = set_time_index(frame, "hour_utc")

    if frame.empty:
        raise ValueError("The snapshot is empty.")

    expected = pd.date_range(
        frame.index.min(),
        frame.index.max(),
        freq="h",
    )

    if len(frame) != 169 or not frame.index.equals(expected):
        raise ValueError("Expected exactly 169 consecutive snapshot hours.")

    frame[METRICS] = frame[METRICS].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(frame[METRICS].to_numpy(dtype=float)).all():
        raise ValueError("Snapshot contains missing or non-finite values.")

    if not frame["weather_source"].eq("open_meteo_gfs_global").all():
        raise ValueError("Expected GFS Global weather inputs.")

    return frame


def download_era5(snapshot, output_dir):
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "start_date": snapshot.index.min().strftime("%Y-%m-%d"),
        "end_date": snapshot.index.max().strftime("%Y-%m-%d"),
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=(15, 120),
    )

    # Preserve the response, including any API error message.
    (output_dir / "era5_response.json").write_text(
        response.text,
        encoding="utf-8",
    )

    save_json(
        output_dir / "request_metadata.json",
        {
            "endpoint": API_URL,
            "parameters": params,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "http_status": response.status_code,
            "snapshot_path": str(SNAPSHOT_PATH),
            "difference_definition": "GFS minus ERA5",
        },
    )

    if not response.ok:
        raise ValueError(
            f"ERA5 request failed: HTTP {response.status_code}. "
            f"Response: {response.text[:1000]}"
        )

    payload = response.json()

    if not isinstance(payload, dict):
        raise TypeError("Expected an API response object.")

    if payload.get("error"):
        raise ValueError(payload.get("reason", "ERA5 API error."))

    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Expected UTC API timestamps.")

    for variable, expected_unit in EXPECTED_UNITS.items():
        actual_unit = payload["hourly_units"].get(variable)

        if actual_unit != expected_unit:
            raise ValueError(
                f"Unexpected unit for {variable}: {actual_unit!r}"
            )

    era5 = set_time_index(
        pd.DataFrame(payload["hourly"]),
        "time",
    )
    era5 = era5[list(VARIABLES)].rename(columns=VARIABLES)
    era5 = era5.apply(pd.to_numeric, errors="raise")

    if np.isinf(era5.to_numpy(dtype=float)).any():
        raise ValueError("ERA5 response contains infinite values.")

    return era5, payload


def compare_metrics(era5, gfs):
    rows = []
    differences = pd.DataFrame(index=gfs.index)

    for metric in METRICS:
        old = era5[metric].to_numpy(dtype=float)
        new = gfs[metric].to_numpy(dtype=float)

        available = np.isfinite(old) & np.isfinite(new)

        if metric == "wind_direction_deg":
            old_speed = era5["wind_speed_ms"].to_numpy(dtype=float)
            new_speed = gfs["wind_speed_ms"].to_numpy(dtype=float)
            available &= np.isfinite(old_speed) & np.isfinite(new_speed)

            calm = available & ((old_speed <= 0) | (new_speed <= 0))
        else:
            calm = np.zeros(len(gfs), dtype=bool)

        usable = available & ~calm
        delta = np.full(len(gfs), np.nan)

        if metric == "wind_direction_deg":
            delta[usable] = (
                new[usable] - old[usable] + 180
            ) % 360 - 180
        else:
            delta[usable] = new[usable] - old[usable]

        differences[metric] = delta
        values = delta[usable]
        count = len(values)

        rows.append(
            {
                "metric": metric,
                "hours_compared": count,
                "unavailable_hours": int((~available).sum()),
                "excluded_zero_wind_hours": int(calm.sum()),
                "mean_signed_difference": (
                    float(values.mean()) if count else np.nan
                ),
                "mean_absolute_difference": (
                    float(np.abs(values).mean()) if count else np.nan
                ),
                "root_mean_square_difference": (
                    float(np.sqrt(np.mean(values**2)))
                    if count else np.nan
                ),
                "largest_absolute_difference": (
                    float(np.abs(values).max()) if count else np.nan
                ),
            }
        )

    return pd.DataFrame(rows), differences


def save_daily_comparison(differences, output_dir):
    local = differences.copy()
    local.index = local.index.tz_convert("Asia/Karachi")

    parts = []

    for metric in METRICS:
        daily = (
            local[metric]
            .abs()
            .resample("D")
            .agg(["count", "mean", "max"])
            .rename(
                columns={
                    "count": "hours_compared",
                    "mean": "mean_absolute_difference",
                    "max": "largest_absolute_difference",
                }
            )
        )

        daily.index.name = "day_pkt"
        daily = daily.reset_index()
        daily["metric"] = metric
        parts.append(daily)

    pd.concat(parts, ignore_index=True).to_csv(
        output_dir / "daily_comparison.csv",
        index=False,
    )


def main():
    snapshot = load_snapshot()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output_dir = (
        BASE_DIR
        / "data/evaluation"
        / f"source_comparison_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    print("AgentWeather — Extended Source Comparison")
    print("Saved GFS window start:", snapshot.index.min())
    print("Saved GFS window end:", snapshot.index.max())
    print("Required hourly rows:", len(snapshot))
    print("Output folder:", output_dir)
    print("\nDownloading ERA5 for the same dates...", flush=True)

    era5, payload = download_era5(snapshot, output_dir)

    missing_timestamps = snapshot.index.difference(era5.index)

    # Match exact timestamps. Unavailable ERA5 values remain missing.
    era5 = era5.reindex(snapshot.index)
    gfs = snapshot[METRICS]

    print(
        "\nERA5 grid:",
        payload.get("latitude"),
        payload.get("longitude"),
    )
    print(
        "GFS grid:",
        snapshot[
            ["weather_grid_latitude", "weather_grid_longitude"]
        ].drop_duplicates().to_dict("records"),
    )

    print("\nMissing ERA5 timestamp rows:", len(missing_timestamps))
    print("Unavailable ERA5 values:")
    print(era5.isna().sum().to_string())

    complete = era5.notna().all(axis=1)
    print(
        "\nHours with all six ERA5 weather values:",
        int(complete.sum()),
    )

    if complete.any():
        available_times = era5.index[complete]
        print("First complete hour:", available_times.min())
        print("Last complete hour:", available_times.max())

    summary, differences = compare_metrics(era5, gfs)

    print("\nSource differences — GFS minus ERA5:")
    print(summary.round(4).to_string(index=False))

    paired = pd.concat(
        [
            era5.add_suffix("_era5"),
            gfs.add_suffix("_gfs"),
            differences.add_suffix("_difference"),
        ],
        axis=1,
    )

    paired.to_csv(output_dir / "hourly_comparison.csv")
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_daily_comparison(differences, output_dir)

    print("\nRain and precipitation coverage:")

    for metric in ("rain_mm", "precipitation_mm"):
        usable = era5[metric].notna()

        print(f"\n{metric}:")
        print("  Comparable hours:", int(usable.sum()))
        print(
            "  ERA5 hours with at least 0.1 mm:",
            int(era5.loc[usable, metric].ge(0.1).sum()),
        )
        print(
            "  GFS hours with at least 0.1 mm:",
            int(gfs.loc[usable, metric].ge(0.1).sum()),
        )

    print("\nSaved comparison to:", output_dir)
    print("Unavailable ERA5 hours were excluded, not filled.")
    print("Wind-direction differences use the shortest angular distance.")
    print("Low-speed wind directions may be unstable.")
    print("These are source differences, not errors against observed truth.")
    print("No model compatibility approval or bias correction was applied.")
    print("Training files, models and dashboard outputs are unchanged.")
    print("No forecasts generated or Hopsworks uploads.")

    if not summary["hours_compared"].gt(0).any():
        print("\nNo usable ERA5 overlap was available.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"\nComparison failed: {error}")
        exit_code = 1
    except requests.RequestException as error:
        print(f"\nAPI connection failed: {error}")
        exit_code = 1

    raise SystemExit(exit_code)