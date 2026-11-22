"""Inspect stored observation history without changing data."""

import os
from pathlib import Path

import hopsworks
import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

SOURCES = {
    "weather": {
        "group": "karachi_weather_observations",
        "time": "observed_at_utc",
        "metrics": [
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
            "wind_direction_deg",
            "rain_api_1h",
            "snow_api_1h",
        ],
    },
    "air_quality": {
        "group": "karachi_air_quality_estimates",
        "time": "valid_time_utc",
        "metrics": [
            "us_aqi",
            "pm2_5_ug_m3",
            "pm10_ug_m3",
            "uv_index",
        ],
    },
}


def prepare_records(frame, specification):
    frame = frame.copy()
    time_column = specification["time"]

    required = [
        "record_id",
        time_column,
        "retrieved_at_utc",
        *specification["metrics"],
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if frame["record_id"].isna().any():
        raise ValueError("Missing record IDs.")

    if frame["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs.")

    for column in (time_column, "retrieved_at_utc"):
        if (
            not frame.empty
            and pd.api.types.is_numeric_dtype(frame[column])
        ):
            raise ValueError(
                f"{column} returned as numeric. "
                "Its timestamp unit must be verified before conversion."
            )

        frame[column] = pd.to_datetime(
            frame[column],
            utc=True,
            errors="raise",
        )

    if frame[time_column].isna().any():
        raise ValueError(f"Missing observation timestamps: {time_column}")

    for column in specification["metrics"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame["hour_utc"] = frame[time_column].dt.floor("h")
    return frame


def inspect_source(name, frame, specification, expected, checked_at):
    time_column = specification["time"]

    print(f"\n{name.upper()}")
    print(f"Stored records: {len(frame)}")

    if not frame.empty:
        latest = frame[time_column].max()
        age = (checked_at - latest).total_seconds() / 3600

        print(f"First observation: {frame[time_column].min()}")
        print(f"Latest observation: {latest}")
        print(f"Latest observation age: {age:.2f} hours")
        print(f"Latest retrieval: {frame['retrieved_at_utc'].max()}")
        print(
            "Records missing retrieval time:",
            int(frame["retrieved_at_utc"].isna().sum()),
        )
        print(
            "Observation timestamps later than this check:",
            int(frame[time_column].gt(checked_at).sum()),
        )

    recent = frame.loc[frame["hour_utc"].isin(expected)].copy()
    present_hours = pd.DatetimeIndex(recent["hour_utc"].unique())
    missing_hours = expected.difference(present_hours)

    print(f"Records in recent window: {len(recent)}")
    print(f"Hours represented: {len(present_hours)}/{len(expected)}")
    print(f"Missing hours: {len(missing_hours)}")
    print(
        "Additional records within represented hours:",
        len(recent) - len(present_hours),
    )

    if len(missing_hours):
        print("First missing hours:")
        for timestamp in missing_hours[:10]:
            print(f"  {timestamp}")
        if len(missing_hours) > 10:
            print(f"  ... and {len(missing_hours) - 10} more")

    # Diagnostic selection only. Do not create model inputs here.
    hourly = (
        recent.sort_values(
            [time_column, "retrieved_at_utc", "record_id"],
            na_position="first",
        )
        .drop_duplicates("hour_utc", keep="last")
        .set_index("hour_utc")
    )

    print("\nInvalid values in selected existing hourly rows:")
    for column in specification["metrics"]:
        values = hourly[column].to_numpy(dtype=float, na_value=np.nan)
        print(f"{column}: {int((~np.isfinite(values)).sum())}")

    print(
        "Selected rows missing retrieval time:",
        int(hourly["retrieved_at_utc"].isna().sum()),
    )
    print("Missing hours are reported separately from invalid values.")

    if not frame.empty:
        print("\nLatest five stored records:")
        columns = [
            time_column,
            "retrieved_at_utc",
            *specification["metrics"],
        ]
        print(
            frame.sort_values(time_column)
            .tail(5)[columns]
            .to_string(index=False)
        )

    return present_hours


def main():
    load_dotenv(BASE_DIR / ".env")

    required = (
        "HOPSWORKS_API_KEY",
        "HOPSWORKS_PROJECT",
        "HOPSWORKS_HOST",
    )
    settings = {
        name: os.getenv(name, "").strip()
        for name in required
    }

    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ValueError("Missing settings: " + ", ".join(missing))

    print("AgentWeather — Stored History Inspection")

    project = hopsworks.login(
        host=settings["HOPSWORKS_HOST"],
        port=443,
        project=settings["HOPSWORKS_PROJECT"],
        api_key_value=settings["HOPSWORKS_API_KEY"],
        engine="python",
    )

    fs = project.get_feature_store()
    frames = {}

    for name, specification in SOURCES.items():
        print(f"\nReading {specification['group']}...", flush=True)
        group = fs.get_feature_group(
            name=specification["group"],
            version=1,
        )
        frames[name] = prepare_records(group.read(), specification)

    checked_at = pd.Timestamp.now(tz="UTC")

    # Exclude the currently unfinished clock hour.
    window_end = checked_at.floor("h") - pd.Timedelta(hours=1)
    expected = pd.date_range(end=window_end, periods=169, freq="h")

    print(f"\nChecked at: {checked_at}")
    print(f"Window start: {expected.min()}")
    print(f"Window end: {expected.max()}")
    print(f"Required hourly buckets: {len(expected)}")

    coverage = {}
    for name, specification in SOURCES.items():
        coverage[name] = inspect_source(
            name,
            frames[name],
            specification,
            expected,
            checked_at,
        )

    both = coverage["weather"].intersection(coverage["air_quality"])

    print("\nCOMBINED COVERAGE")
    print(f"Hours represented by both sources: {len(both)}/{len(expected)}")
    print(f"Hours lacking one or both sources: {len(expected) - len(both)}")

    print("\nCoverage alone does not establish forecast readiness.")
    print("Hour labels do not establish when observations became available.")
    print("Rainfall mapping and training/live source compatibility remain unresolved.")
    print("No files saved, values filled, models run or data uploaded.")


if __name__ == "__main__":
    main()