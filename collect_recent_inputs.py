"""Save recent model-estimate inputs for compatibility evaluation."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
LATITUDE = 24.8607
LONGITUDE = 67.0011
WINDOW_HOURS = 169

WEATHER_VARIABLES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precipitation_mm",
    "rain": "rain_mm",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
}

AIR_VARIABLES = {
    "us_aqi": "us_aqi",
    "pm2_5": "pm2_5_ug_m3",
    "pm10": "pm10_ug_m3",
    "uv_index": "uv_index",
}

WEATHER_UNITS = {
    "temperature_2m": {"°C"},
    "relative_humidity_2m": {"%"},
    "precipitation": {"mm"},
    "rain": {"mm"},
    "wind_speed_10m": {"m/s"},
    "wind_direction_10m": {"°"},
}

AIR_UNITS = {
    "us_aqi": {"USAQI"},
    "pm2_5": {"μg/m³", "µg/m³"},
    "pm10": {"μg/m³", "µg/m³"},
    "uv_index": {""},
}


def save_json(path, content):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            content,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def fetch_source(url, params, raw_path):
    requested_at = datetime.now(UTC)

    response = requests.get(
        url,
        params=params,
        timeout=(15, 120),
    )
    response.raise_for_status()

    retrieved_at = datetime.now(UTC)
    payload = response.json()

    # Preserve the response before validation for troubleshooting.
    save_json(raw_path, payload)

    if not isinstance(payload, dict):
        raise TypeError("Expected a single-location API response.")

    if payload.get("error"):
        raise ValueError(payload.get("reason", "API returned an error."))

    metadata = {
        "endpoint": url,
        "parameters": params,
        "requested_at_utc": requested_at.isoformat(),
        "retrieved_at_utc": retrieved_at.isoformat(),
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "elevation": payload.get("elevation"),
        "hourly_units": payload.get("hourly_units"),
    }

    return payload, metadata


def validate_source(payload, mapping, expected_units, expected):
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Expected UTC timestamps with zero offset.")

    for coordinate in ("latitude", "longitude"):
        value = float(payload[coordinate])
        if not np.isfinite(value):
            raise ValueError(f"Invalid grid coordinate: {coordinate}")

    units = payload["hourly_units"]
    for variable, allowed in expected_units.items():
        if units.get(variable) not in allowed:
            raise ValueError(
                f"Unexpected unit for {variable}: {units.get(variable)!r}"
            )

    frame = pd.DataFrame(payload["hourly"])

    missing_columns = {"time", *mapping} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing variables: {sorted(missing_columns)}")

    frame["time"] = pd.to_datetime(
        frame["time"],
        utc=True,
        errors="raise",
    )

    if frame["time"].isna().any():
        raise ValueError("Missing timestamps.")

    if frame["time"].duplicated().any():
        raise ValueError("Duplicate timestamps.")

    if not frame["time"].eq(frame["time"].dt.floor("h")).all():
        raise ValueError("Expected exact hourly timestamps.")

    frame = frame.set_index("time").sort_index()

    missing_hours = expected.difference(frame.index)
    if len(missing_hours):
        raise ValueError(
            f"Missing {len(missing_hours)} required hours. "
            f"First missing hour: {missing_hours[0]}"
        )

    # Select only the fixed input window, excluding future target hours.
    window = frame.loc[expected, list(mapping)].copy()

    for column in mapping:
        window[column] = pd.to_numeric(window[column], errors="raise")

    if not np.isfinite(window.to_numpy(dtype=float)).all():
        raise ValueError("Required window contains missing or infinite values.")

    window = window.rename(columns=mapping)
    window.index.name = "hour_utc"
    return window


def validate_combined(frame):
    if not frame["temperature_c"].between(-100, 65).all():
        raise ValueError("Temperature failed the broad physical-range check.")

    if not frame["humidity_pct"].between(0, 100).all():
        raise ValueError("Humidity must be between 0 and 100.")

    if not frame["wind_direction_deg"].between(0, 360).all():
        raise ValueError("Wind direction must be between 0 and 360.")

    nonnegative = [
        "precipitation_mm",
        "rain_mm",
        "wind_speed_ms",
        "us_aqi",
        "pm2_5_ug_m3",
        "pm10_ug_m3",
        "uv_index",
    ]
    if frame[nonnegative].lt(0).any().any():
        raise ValueError("Unexpected negative metric values.")

    # These are broad validity checks, not evidence of source accuracy.


def main():
    started_at = datetime.now(UTC)
    snapshot_id = (
        started_at.strftime("%Y%m%dT%H%M%S%f")
        + "_"
        + uuid4().hex[:8]
    )
    output_dir = DATA_DIR = BASE_DIR / "data" / "recent_inputs"
    output_dir = DATA_DIR / snapshot_id
    output_dir.mkdir(parents=True, exist_ok=False)

    cutoff = (
        pd.Timestamp(started_at).floor("h")
        - pd.Timedelta(hours=1)
    )
    expected = pd.date_range(
        end=cutoff,
        periods=WINDOW_HOURS,
        freq="h",
    )

    manifest = {
        "snapshot_id": snapshot_id,
        "status": "started",
        "purpose": "recent_input_compatibility_evaluation",
        "started_at_utc": started_at.isoformat(),
        "requested_latitude": LATITUDE,
        "requested_longitude": LONGITUDE,
        "window_start_utc": expected.min().isoformat(),
        "window_end_utc": expected.max().isoformat(),
        "required_rows": WINDOW_HOURS,
        "sources": {},
        "limitations": [
            "Inputs are weather-model and air-quality model estimates.",
            "Retrieval time is not the model initialization time.",
            "Past values retrieved now are not an as-issued historical archive.",
            "Existing ERA5-trained models are not validated for these GFS inputs.",
            "This snapshot alone cannot reconstruct the existing SARIMAX state.",
        ],
    }

    print("AgentWeather — Recent Input Snapshot")
    print(f"Window start: {expected.min()}")
    print(f"Window end: {expected.max()}")
    print(f"Required rows: {WINDOW_HOURS}")

    try:
        weather_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(WEATHER_VARIABLES),
            "models": "gfs_global",
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "past_days": 8,
            "forecast_days": 1,
        }

        print("\nDownloading GFS weather estimates...", flush=True)
        weather_payload, weather_metadata = fetch_source(
            "https://api.open-meteo.com/v1/forecast",
            weather_params,
            output_dir / "weather_response.json",
        )
        manifest["sources"]["weather"] = weather_metadata

        weather = validate_source(
            weather_payload,
            WEATHER_VARIABLES,
            WEATHER_UNITS,
            expected,
        )

        air_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(AIR_VARIABLES),
            "domains": "cams_global",
            "timezone": "UTC",
            "start_date": expected.min().strftime("%Y-%m-%d"),
            "end_date": expected.max().strftime("%Y-%m-%d"),
        }

        print("Downloading CAMS air-quality estimates...", flush=True)
        air_payload, air_metadata = fetch_source(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            air_params,
            output_dir / "air_quality_response.json",
        )
        manifest["sources"]["air_quality"] = air_metadata

        air = validate_source(
            air_payload,
            AIR_VARIABLES,
            AIR_UNITS,
            expected,
        )

        combined = weather.join(air, how="inner", validate="one_to_one")

        if not combined.index.equals(expected):
            raise ValueError("Combined timestamps do not match the required window.")

        validate_combined(combined)
        metric_columns = list(combined.columns)

        combined["weather_source"] = "open_meteo_gfs_global"
        combined["weather_data_type"] = "weather_model_estimate"
        combined["air_source"] = "open_meteo_cams_global"
        combined["air_data_type"] = "modelled_estimate"

        for prefix, metadata in (
            ("weather", weather_metadata),
            ("air", air_metadata),
        ):
            combined[f"{prefix}_retrieved_at_utc"] = metadata["retrieved_at_utc"]
            combined[f"{prefix}_grid_latitude"] = metadata["grid_latitude"]
            combined[f"{prefix}_grid_longitude"] = metadata["grid_longitude"]

        combined["requested_latitude"] = LATITUDE
        combined["requested_longitude"] = LONGITUDE
        combined["snapshot_id"] = snapshot_id

        csv_path = output_dir / "recent_hourly_inputs.csv"
        temporary = output_dir / "recent_hourly_inputs.tmp"
        combined.to_csv(temporary, index_label="hour_utc")
        temporary.replace(csv_path)

        manifest["status"] = "validated"
        manifest["rows"] = len(combined)
        manifest["metric_columns"] = metric_columns
        manifest["combined_file"] = csv_path.name
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()

        save_json(output_dir / "manifest.json", manifest)

        print(f"\nValidated rows: {len(combined)}")
        print(f"Metric columns: {len(metric_columns)}")
        print("Missing hours: 0")
        print("Missing metric values: 0")
        print("\nLatest five rows:")
        print(combined[metric_columns].tail(5).round(3).to_string())

        print(f"\nSnapshot saved to: {output_dir}")
        print("Raw API responses and source metadata preserved.")
        print("Snapshot is for compatibility evaluation, not approved model input.")
        print("No training files, models or dashboard outputs changed.")
        print("No Hopsworks upload or forecasts generated.")
        return 0

    except (
        requests.RequestException,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()

        try:
            save_json(output_dir / "failure.json", manifest)
        except OSError:
            print("Could not save the failure report.")

        print(f"\nCollection failed: {error}")
        print(f"Any downloaded responses remain in: {output_dir}")
        print("Do not use this failed snapshot for forecasting.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())