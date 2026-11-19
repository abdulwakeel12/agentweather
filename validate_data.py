import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "hourly_dataset.csv"

# Basic consistency checks, not hazard-alert thresholds.
NUMERIC_RULES = {
    "temperature_c": (None, None),
    "humidity_pct": (0, 100),
    "wind_speed_ms": (0, None),
    "wind_direction_deg": (0, 360),
    "rain_api_1h": (0, None),
    "snow_api_1h": (0, None),
    "us_aqi": (0, None),
    "pm2_5_ug_m3": (0, None),
    "pm10_ug_m3": (0, None),
    "uv_index": (0, None),
}


def parse_time(value):
    timestamp = datetime.fromisoformat(value)

    if timestamp.tzinfo is None:
        raise ValueError("Timestamp has no timezone.")

    return timestamp.astimezone(UTC)


def main():
    if not DATA_PATH.exists():
        print("Run run_pipeline.py first.")
        raise SystemExit(1)

    try:
        with DATA_PATH.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames or []
            rows = list(reader)

        required = {
            "hour_utc",
            "weather_available",
            "air_available",
            *NUMERIC_RULES,
        }

        missing_columns = required - set(columns)
        if missing_columns:
            print("Missing columns:", ", ".join(sorted(missing_columns)))
            raise SystemExit(1)

        if not rows:
            print("The dataset is empty.")
            raise SystemExit(1)

        errors = []
        hours = []
        missing_values = {column: 0 for column in NUMERIC_RULES}

        for line_number, row in enumerate(rows, start=2):
            try:
                hour = parse_time(row["hour_utc"])
                hours.append(hour)

                if hour.minute or hour.second or hour.microsecond:
                    errors.append(f"Line {line_number}: hour is not rounded.")
            except (ValueError, TypeError):
                errors.append(f"Line {line_number}: invalid hour timestamp.")

            for flag in ("weather_available", "air_available"):
                if row.get(flag) not in ("True", "False"):
                    errors.append(f"Line {line_number}: invalid {flag}.")

            for column, (minimum, maximum) in NUMERIC_RULES.items():
                raw = row.get(column)

                if raw is None or not raw.strip():
                    missing_values[column] += 1
                    continue

                try:
                    value = float(raw)

                    if not math.isfinite(value):
                        raise ValueError

                    if minimum is not None and value < minimum:
                        raise ValueError

                    if maximum is not None and value > maximum:
                        raise ValueError

                except ValueError:
                    errors.append(
                        f"Line {line_number}: invalid {column} value."
                    )

        unique_hours = sorted(set(hours))
        duplicates = len(hours) - len(unique_hours)

        if duplicates:
            errors.append(f"Duplicate hourly rows: {duplicates}")

        print("\nAgentWeather — Local Data Check")
        print(f"Rows: {len(rows)}")
        print(f"Duplicate hours: {duplicates}")

        if unique_hours:
            first = unique_hours[0]
            last = unique_hours[-1]
            expected_count = int(
                (last - first).total_seconds() // 3600
            ) + 1
            missing_hours = expected_count - len(unique_hours)

            print(f"First hour: {first.isoformat()}")
            print(f"Latest hour: {last.isoformat()}")
            print(f"Hours absent between first and latest: {missing_hours}")

            if missing_hours:
                print("\nCollection gaps:")
                for previous, current in zip(
                    unique_hours, unique_hours[1:]
                ):
                    gap = int(
                        (current - previous).total_seconds() // 3600
                    ) - 1

                    if gap > 0:
                        start = previous + timedelta(hours=1)
                        end = current - timedelta(hours=1)
                        print(
                            f"  {start.isoformat()} to "
                            f"{end.isoformat()}: {gap} missing hour(s)"
                        )

        incomplete = sum(
            row["weather_available"] != "True"
            or row["air_available"] != "True"
            for row in rows
        )

        print(f"\nExisting rows missing a source: {incomplete}")
        print("\nBlank values by column:")

        for column, count in missing_values.items():
            print(f"  {column}: {count}")

        if errors:
            print("\nValidation errors:")
            for error in errors:
                print(f"  {error}")
            raise SystemExit(1)

        print("\nBasic validation passed.")
        print("Gaps and blank values still need a documented strategy.")
        print("No files were changed.")

    except (OSError, csv.Error, UnicodeError):
        print("Could not read the CSV. Check the file and retry.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()