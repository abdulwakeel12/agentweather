from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DAY = "2025-11-15"

SOURCE_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "weather_era5_2025-11-01_2025-11-30.csv"
)
OUTPUT_DIR = BASE_DIR / "data" / "diagnostics"

COLUMNS = ["temperature_c", "humidity_pct"]


def main():
    try:
        saved = pd.read_csv(SOURCE_PATH)
        saved["hour_utc"] = pd.to_datetime(
            saved["hour_utc"], utc=True, errors="raise"
        )

        start = pd.Timestamp(DAY, tz="UTC")
        end = start + pd.Timedelta(days=1)
        expected = pd.date_range(start, periods=24, freq="h")

        saved = saved.loc[
            (saved["hour_utc"] >= start)
            & (saved["hour_utc"] < end)
        ].sort_values("hour_utc")

        if not pd.DatetimeIndex(saved["hour_utc"]).equals(expected):
            raise ValueError("Saved ERA5 timestamps are incomplete.")

        params = {
            "latitude": 24.8607,
            "longitude": 67.0011,
            "start_date": DAY,
            "end_date": DAY,
            "hourly": "temperature_2m,relative_humidity_2m",
            "models": "era5_land",
            "timezone": "UTC",
            "temperature_unit": "celsius",
        }

        print(f"Comparing saved ERA5 with ERA5-Land for {DAY}...")

        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        if data["utc_offset_seconds"] != 0:
            raise ValueError("Unexpected timezone offset.")

        hourly = data["hourly"]
        land = pd.DataFrame(
            {
                "hour_utc": pd.to_datetime(
                    hourly["time"], utc=True, errors="raise"
                ),
                "temperature_c": hourly["temperature_2m"],
                "humidity_pct": hourly["relative_humidity_2m"],
            }
        ).sort_values("hour_utc")

        if not pd.DatetimeIndex(land["hour_utc"]).equals(expected):
            raise ValueError("ERA5-Land timestamps are incomplete.")

        for frame in (saved, land):
            for column in COLUMNS:
                frame[column] = pd.to_numeric(
                    frame[column], errors="raise"
                )
            if not np.isfinite(
                frame[COLUMNS].to_numpy(dtype=float)
            ).all():
                raise ValueError("Missing or non-finite metric values.")

        comparison = saved[["hour_utc", *COLUMNS]].merge(
            land,
            on="hour_utc",
            validate="one_to_one",
            suffixes=("_era5", "_era5_land"),
        )

        comparison["time_pkt"] = (
            comparison["hour_utc"]
            .dt.tz_convert("Asia/Karachi")
            .dt.strftime("%Y-%m-%d %H:%M")
        )

        print("\nGrid coordinates:")
        print(
            "Saved ERA5:",
            saved[["grid_latitude", "grid_longitude"]]
            .drop_duplicates()
            .to_dict("records"),
        )
        print(
            f"ERA5-Land: latitude={data['latitude']}, "
            f"longitude={data['longitude']}"
        )

        print("\nHourly comparison:")
        print(
            comparison[
                [
                    "time_pkt",
                    "temperature_c_era5",
                    "temperature_c_era5_land",
                    "humidity_pct_era5",
                    "humidity_pct_era5_land",
                ]
            ].to_string(index=False)
        )

        print("\nLargest absolute temperature change between adjacent hours:")
        for model in ("era5", "era5_land"):
            column = f"temperature_c_{model}"
            changes = comparison[column].diff()
            largest = changes.abs().idxmax()
            print(
                f"{model}: {changes.loc[largest]:+.2f} °C "
                f"at {comparison.loc[largest, 'time_pkt']} PKT"
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        output_path = OUTPUT_DIR / f"era5_land_comparison_{DAY}_{run_id}.csv"

        comparison["era5_land_grid_latitude"] = data["latitude"]
        comparison["era5_land_grid_longitude"] = data["longitude"]
        comparison.to_csv(output_path, index=False)

        print(f"\nComparison saved to: {output_path}")
        print("Different grid locations can contribute to differences.")
        print("A smoother series alone does not prove greater accuracy.")
        print("Training data unchanged. No Hopsworks upload.")

    except requests.exceptions.RequestException:
        print("API request failed. Check your connection and retry later.")
        raise SystemExit(1) from None
    except (ValueError, KeyError, TypeError) as error:
        print(f"Comparison failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check file paths and permissions.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()