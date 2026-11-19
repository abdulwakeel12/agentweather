import argparse
import calendar
import time
from pathlib import Path

import backfill_air_quality
import backfill_weather
import build_historical_dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"


def month_range(start_month, end_month):
    """Yield monthly date ranges, including both endpoint months."""
    start_year, start_number = map(int, start_month.split("-"))
    end_year, end_number = map(int, end_month.split("-"))

    if not 1 <= start_number <= 12 or not 1 <= end_number <= 12:
        raise ValueError("Month numbers must be between 01 and 12.")

    if (start_year, start_number) > (end_year, end_number):
        raise ValueError("Start month must come before end month.")

    year, month = start_year, start_number

    while (year, month) <= (end_year, end_number):
        last_day = calendar.monthrange(year, month)[1]
        yield (
            f"{year:04d}-{month:02d}-01",
            f"{year:04d}-{month:02d}-{last_day:02d}",
        )

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def download_source(module, path, force):
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"\nUsing existing file: {path.name}", flush=True)
        return

    module.main()

    # Brief pause between API requests.
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser(
        description="Download and validate historical data month by month."
    )
    parser.add_argument("--start", default="2025-11", help="YYYY-MM")
    parser.add_argument("--end", default="2026-10", help="YYYY-MM")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when source CSVs already exist.",
    )
    args = parser.parse_args()

    try:
        months = list(month_range(args.start, args.end))
    except (ValueError, OverflowError):
        print("Invalid range. Use valid months such as 2025-11 and 2026-10.")
        return 1

    for number, (start_date, end_date) in enumerate(months, start=1):
        print(
            f"\n=== Month {number}/{len(months)}: "
            f"{start_date} to {end_date} ===",
            flush=True,
        )

        # Configure the existing scripts for this month.
        for module in (
            backfill_weather,
            backfill_air_quality,
            build_historical_dataset,
        ):
            module.START_DATE = start_date
            module.END_DATE = end_date

        weather_path = DATA_DIR / (
            f"weather_era5_{start_date}_{end_date}.csv"
        )
        air_path = DATA_DIR / (
            f"air_quality_cams_global_{start_date}_{end_date}.csv"
        )

        try:
            download_source(
                backfill_weather, weather_path, args.force
            )
            download_source(
                backfill_air_quality, air_path, args.force
            )

            # Always validate both sources, including reused CSVs.
            build_historical_dataset.main()

        except SystemExit as error:
            if error.code not in (None, 0):
                print(
                    f"\nStopped at {start_date[:7]}. "
                    "Previously completed months remain saved."
                )
                print("Fix the reported issue, then rerun the command.")
                return 1
            raise

    print(f"\nCompleted and validated {len(months)} month(s).")
    print(f"Files saved in: {DATA_DIR}")
    print("No data uploaded to Hopsworks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())