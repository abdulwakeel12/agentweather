import json
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

VARIABLES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
}


def main():
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "start_date": DAY,
        "end_date": DAY,
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }

    try:
        saved = pd.read_csv(SOURCE_PATH)
        saved["hour_utc"] = pd.to_datetime(
            saved["hour_utc"], utc=True, errors="raise"
        )

        start = pd.Timestamp(DAY, tz="UTC")
        end = start + pd.Timedelta(days=1)

        saved = saved.loc[
            (saved["hour_utc"] >= start)
            & (saved["hour_utc"] < end)
        ].sort_values("hour_utc")

        expected = pd.date_range(start, periods=24, freq="h")

        if not pd.DatetimeIndex(saved["hour_utc"]).equals(expected):
            raise ValueError("Saved data does not contain the expected 24 hours.")

        print(f"Fetching a fresh ERA5 response for {DAY} (UTC)...")

        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        # Preserve the fresh response before performing comparisons.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        raw_path = OUTPUT_DIR / f"era5_response_{DAY}_{run_id}.json"

        raw_path.write_text(
            json.dumps(
                {
                    "retrieved_at_utc": datetime.now(UTC).isoformat(),
                    "request_parameters": params,
                    "response": data,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Fresh response saved to: {raw_path}")

        if data["utc_offset_seconds"] != 0:
            raise ValueError("The response has an unexpected timezone offset.")

        hourly = data["hourly"]
        fresh = pd.DataFrame(
            {
                "hour_utc": pd.to_datetime(
                    hourly["time"], utc=True, errors="raise"
                ),
                **{
                    column: hourly[variable]
                    for variable, column in VARIABLES.items()
                },
            }
        ).sort_values("hour_utc")

        if not pd.DatetimeIndex(fresh["hour_utc"]).equals(expected):
            raise ValueError("Fresh response does not contain the expected hours.")

        print("\nGrid coordinates:")
        grid_matches = True

        for column, response_key in (
            ("grid_latitude", "latitude"),
            ("grid_longitude", "longitude"),
        ):
            old_values = pd.to_numeric(saved[column], errors="raise")
            new_value = float(data[response_key])
            matches = np.isclose(
                old_values.to_numpy(), new_value, rtol=0, atol=1e-6
            ).all()
            grid_matches = grid_matches and bool(matches)

            print(
                f"{column}: saved={old_values.unique().tolist()}, "
                f"fresh={new_value}, match={bool(matches)}"
            )

        comparison = saved[["hour_utc", *VARIABLES.values()]].merge(
            fresh,
            on="hour_utc",
            validate="one_to_one",
            suffixes=("_saved", "_fresh"),
        )

        values_match = True

        print("\nValue comparison:")
        for column in VARIABLES.values():
            old = pd.to_numeric(
                comparison[f"{column}_saved"], errors="raise"
            )
            new = pd.to_numeric(
                comparison[f"{column}_fresh"], errors="raise"
            )

            if not np.isfinite(old).all() or not np.isfinite(new).all():
                raise ValueError(f"Missing or non-finite values in {column}.")

            matches = np.isclose(old, new, rtol=0, atol=1e-9)
            values_match = values_match and bool(matches.all())

            comparison[f"{column}_difference"] = new - old
            print(f"{column}: {matches.sum()}/24 match")
            print(f"Maximum absolute difference: {(new - old).abs().max():.4f}")

        comparison["time_pkt"] = (
            comparison["hour_utc"]
            .dt.tz_convert("Asia/Karachi")
            .dt.strftime("%Y-%m-%d %H:%M")
        )

        print("\nFresh readings in Karachi time:")
        print(
            comparison[
                ["time_pkt", "temperature_c_fresh", "humidity_pct_fresh"]
            ].to_string(index=False)
        )

        comparison_path = OUTPUT_DIR / f"era5_comparison_{DAY}_{run_id}.csv"
        comparison.to_csv(comparison_path, index=False)

        print(f"\nComparison saved to: {comparison_path}")

        if grid_matches and values_match:
            print("Fresh API values and grid coordinates match the saved data.")
            print("This confirms reproducibility, not independent accuracy.")
        else:
            print("Differences found. Review them before changing training data.")

        print("Training files unchanged. No Hopsworks upload.")

    except requests.exceptions.RequestException:
        print("API request failed. Check your connection and retry later.")
        raise SystemExit(1) from None
    except (ValueError, KeyError, TypeError) as error:
        print(f"Verification failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and folder permissions.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()