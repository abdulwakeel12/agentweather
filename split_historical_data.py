import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "historical_hourly_2025-11_2026-10.csv"
)
OUTPUT_DIR = BASE_DIR / "data" / "splits"

START = datetime(2025, 11, 1, tzinfo=UTC)
VALIDATION_START = datetime(2026, 8, 1, tzinfo=UTC)
TEST_START = datetime(2026, 10, 1, tzinfo=UTC)
END = datetime(2026, 11, 1, tzinfo=UTC)


def main():
    splits = {"train": [], "validation": [], "test": []}

    try:
        with INPUT_PATH.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames

            if not fields or "hour_utc" not in fields:
                raise ValueError("Missing hour_utc column.")

            rows = list(reader)

        expected_time = START

        for row in rows:
            timestamp = datetime.fromisoformat(row["hour_utc"])

            if timestamp.tzinfo is None:
                raise ValueError("A timestamp has no timezone.")

            timestamp = timestamp.astimezone(UTC)

            if timestamp != expected_time or timestamp >= END:
                raise ValueError(
                    "Unexpected timestamp order, duplicate or missing hour."
                )

            if timestamp < VALIDATION_START:
                splits["train"].append(row)
            elif timestamp < TEST_START:
                splits["validation"].append(row)
            else:
                splits["test"].append(row)

            expected_time += timedelta(hours=1)

        if expected_time != END:
            raise ValueError("The input does not cover the complete year.")

        if any(not records for records in splits.values()):
            raise ValueError("One or more splits are empty.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for name, records in splits.items():
            output_path = OUTPUT_DIR / f"{name}.csv"
            temp_path = output_path.with_suffix(".tmp")

            with temp_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)

            temp_path.replace(output_path)

            print(f"\n{name.upper()}")
            print(f"Rows: {len(records)}")
            print(f"First hour: {records[0]['hour_utc']}")
            print(f"Last hour: {records[-1]['hour_utc']}")

    except (ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"Split failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and close open CSV files.")
        raise SystemExit(1) from None

    print(f"\nSplit files saved to: {OUTPUT_DIR}")
    print("Original annual dataset preserved.")
    print("No shuffling, imputation or scaling performed.")
    print("No data uploaded to Hopsworks.")


if __name__ == "__main__":
    main()