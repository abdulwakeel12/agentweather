from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "historical_hourly_2025-11_2026-10.csv"
)
FEATURE_DIR = BASE_DIR / "data" / "features"
OUTPUT_DIR = BASE_DIR / "data" / "supervised"

HORIZONS = range(1, 25)
RAIN_THRESHOLD_MM = 0.1

SPLITS = {
    "train": ("2025-11-08", "2026-08-01"),
    "validation": ("2026-08-01", "2026-10-01"),
    "test": ("2026-10-01", "2026-11-01"),
}

REQUIRED = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "rain_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
    "wind_direction_deg",
]


def read_indexed_csv(path):
    df = pd.read_csv(path)
    df["hour_utc"] = pd.to_datetime(
        df["hour_utc"], utc=True, errors="raise"
    )

    if df["hour_utc"].isna().any() or df["hour_utc"].duplicated().any():
        raise ValueError(f"Invalid or duplicate timestamps in {path.name}.")

    return df.set_index("hour_utc").sort_index()


def write_csv(df, path):
    temporary = path.with_suffix(".tmp")
    df.to_csv(temporary, index_label="hour_utc")
    temporary.replace(path)


def main():
    try:
        history = read_indexed_csv(HISTORY_PATH)

        expected_history = pd.date_range(
            "2025-11-01",
            "2026-11-01",
            inclusive="left",
            freq="h",
            tz="UTC",
        )
        if not history.index.equals(expected_history):
            raise ValueError("Historical timestamps are incomplete.")

        for column in REQUIRED:
            history[column] = pd.to_numeric(
                history[column], errors="raise"
            )

        if not np.isfinite(
            history[REQUIRED].to_numpy(dtype=float)
        ).all():
            raise ValueError("Missing or non-finite historical values.")

        # Values to predict at each future hour.
        outcomes = history[
            [
                "us_aqi",
                "temperature_c",
                "precipitation_mm",
                "humidity_pct",
                "uv_index",
                "wind_speed_ms",
            ]
        ].copy()

        outcomes["rain_event"] = (
            history["rain_mm"] >= RAIN_THRESHOLD_MM
        ).astype(int)

        # Encode direction as two circular components.
        # Direction is undefined at zero wind speed.
        moving = history["wind_speed_ms"].gt(0)
        radians = np.deg2rad(history["wind_direction_deg"] % 360)

        outcomes["wind_direction_sin"] = np.sin(radians).where(moving, 0.0)
        outcomes["wind_direction_cos"] = np.cos(radians).where(moving, 0.0)

        # This mask is for evaluating direction, not a prediction target.
        direction_valid = moving.astype(int)

        target_columns = {}
        mask_columns = {}

        for horizon in HORIZONS:
            for column in outcomes.columns:
                target_columns[f"{column}_t_plus_{horizon}h"] = (
                    outcomes[column].shift(-horizon)
                )

            mask_columns[f"wind_direction_valid_t_plus_{horizon}h"] = (
                direction_valid.shift(-horizon)
            )

        targets = pd.DataFrame(target_columns, index=history.index)
        masks = pd.DataFrame(mask_columns, index=history.index)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for name, (start_text, end_text) in SPLITS.items():
            start = pd.Timestamp(start_text, tz="UTC")
            end = pd.Timestamp(end_text, tz="UTC")

            features = read_indexed_csv(
                FEATURE_DIR / f"{name}_features.csv"
            )

            expected_features = pd.date_range(
                start, end, inclusive="left", freq="h"
            )
            if not features.index.equals(expected_features):
                raise ValueError(f"Unexpected timestamps in {name} features.")

            # All 24 future hours must stay inside this split.
            keep = features.index + pd.Timedelta(hours=24) < end
            x = features.loc[keep].copy()
            y = targets.loc[x.index].copy()
            direction_mask = masks.loc[x.index].copy()

            if x.empty:
                raise ValueError(f"No usable examples in {name}.")

            for frame in (x, y, direction_mask):
                if not np.isfinite(
                    frame.to_numpy(dtype=float)
                ).all():
                    raise ValueError(f"Invalid values in {name} output.")

            direction_mask = direction_mask.astype(int)

            # Keep future outcomes separate from model inputs.
            write_csv(x, OUTPUT_DIR / f"{name}_X.csv")
            write_csv(y, OUTPUT_DIR / f"{name}_y.csv")
            write_csv(
                direction_mask,
                OUTPUT_DIR / f"{name}_direction_mask.csv",
            )

            print(f"\n{name.upper()}")
            print(f"Input feature rows: {len(features)}")
            print(f"Usable examples: {len(x)}")
            print(f"Boundary rows excluded: {len(features) - len(x)}")
            print(f"Input features: {len(x.columns)}")
            print(f"Target columns: {len(y.columns)}")
            print(f"First prediction time: {x.index.min()}")
            print(f"Last prediction time: {x.index.max()}")
            print(
                "Latest target time: "
                f"{x.index.max() + pd.Timedelta(hours=24)}"
            )

        print(f"\nSaved to: {OUTPUT_DIR}")
        print("Forecast horizons: 1–24 hours ahead.")
        print(f"Rain event definition: rain_mm >= {RAIN_THRESHOLD_MM}")
        print("Direction masks are for evaluation, not input features.")
        print("Original files unchanged. No Hopsworks upload.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"Target generation failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and close open output files.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()