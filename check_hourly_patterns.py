from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "splits" / "train.csv"
HISTORY_DIR = BASE_DIR / "data" / "historical"

COLUMNS = ["temperature_c", "humidity_pct"]


def read_data(path):
    df = pd.read_csv(path)
    df["hour_utc"] = pd.to_datetime(
        df["hour_utc"], utc=True, errors="raise"
    )

    if df["hour_utc"].isna().any() or df["hour_utc"].duplicated().any():
        raise ValueError(f"Missing or duplicate timestamps in {path.name}.")

    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(df[COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or invalid numeric values in {path.name}.")

    return df.set_index("hour_utc").sort_index()


def main():
    try:
        train = read_data(TRAIN_PATH)

        expected = pd.date_range(
            "2025-11-01",
            "2026-08-01",
            inclusive="left",
            freq="h",
            tz="UTC",
        )
        if not train.index.equals(expected):
            raise ValueError("Unexpected training timestamps.")

        source_frames = []

        # Load only the nine training months, excluding the sample week.
        for month in pd.period_range("2025-11", "2026-07", freq="M"):
            start = month.start_time.strftime("%Y-%m-%d")
            end = month.end_time.strftime("%Y-%m-%d")
            path = HISTORY_DIR / f"weather_era5_{start}_{end}.csv"
            source_frames.append(read_data(path)[COLUMNS])

        source = pd.concat(source_frames).sort_index()

        if not source.index.equals(train.index):
            raise ValueError("Source and training timestamps do not match.")

        print("\nComparison with downloaded weather CSVs:")

        all_match = True

        for column in COLUMNS:
            matches = np.isclose(
                train[column].to_numpy(),
                source[column].to_numpy(),
                rtol=0,
                atol=1e-9,
            )
            print(f"{column}: {matches.sum()}/{len(matches)} match")
            all_match = all_match and bool(matches.all())

        # Compute changes in chronological order before grouping by hour.
        changes = train[COLUMNS].diff()
        changes.columns = [
            "temperature_change_c",
            "humidity_change_percentage_points",
        ]

        local_hours = changes.index.tz_convert("Asia/Karachi").hour
        hourly_changes = changes.groupby(local_hours).mean()
        hourly_changes.index.name = "hour_pkt"

        print("\nAverage change from the previous hour:")
        print(hourly_changes.round(3).to_string())

        # Show individual days from different training months.
        local = train[COLUMNS].copy()
        local.index = local.index.tz_convert("Asia/Karachi")

        for day in ["2025-11-15", "2026-03-15", "2026-07-15"]:
            sample = local.loc[day].copy()
            sample.index = sample.index.strftime("%H:%M")
            sample.index.name = "time_pkt"
            print(f"\nIndividual day: {day} — Karachi time")
            print(sample.round(2).to_string())

        if all_match:
            print(
                "\nBoth columns match the downloaded source CSVs."
                "\nTheir values were not changed by merging or splitting."
                "\nThis does not independently verify the API data."
            )
        else:
            print("\nValues differ. Investigate processing before training.")
            raise SystemExit(1)

        print("\nNo files changed. No API calls or Hopsworks uploads.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"Check failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("Could not read the CSV files. Check their locations.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()