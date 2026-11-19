import math
import os
from pathlib import Path

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from prepare_data import prepare_source
from upload_weather import (
    BOOLEAN_COLUMNS,
    NUMERIC_COLUMNS,
    TIMESTAMP_COLUMNS,
    make_record_id,
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def values_match(left, right, numeric=False):
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)

    if left_missing or right_missing:
        return left_missing and right_missing

    if numeric:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    return left == right


def main():
    local = prepare_source("weather")
    local.insert(0, "record_id", local.apply(make_record_id, axis=1))

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
        print("Missing .env settings:", ", ".join(missing))
        raise SystemExit(1)

    project = hopsworks.login(
        host=settings["HOPSWORKS_HOST"],
        port=443,
        project=settings["HOPSWORKS_PROJECT"],
        api_key_value=settings["HOPSWORKS_API_KEY"],
        engine="python",
    )

    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name="karachi_weather_observations",
        version=1,
    )

    print("\nReading weather records from Hopsworks...")
    remote = fg.read()

    print(f"Local records: {len(local)}")
    print(f"Hopsworks records: {len(remote)}")

    missing_columns = set(local.columns) - set(remote.columns)
    if missing_columns:
        print("Missing remote columns:", sorted(missing_columns))
        raise SystemExit(1)

    if remote["record_id"].isna().any():
        print("Missing record IDs found in Hopsworks.")
        raise SystemExit(1)

    if remote["record_id"].duplicated().any():
        print("Duplicate record IDs found in Hopsworks.")
        raise SystemExit(1)

    for column in TIMESTAMP_COLUMNS:
        if pd.api.types.is_numeric_dtype(remote[column]):
            print(f"{column} returned as numeric; timestamp unit needs checking.")
            raise SystemExit(1)

        remote[column] = pd.to_datetime(
            remote[column],
            utc=True,
            errors="raise",
        )

    for column in NUMERIC_COLUMNS:
        remote[column] = pd.to_numeric(remote[column], errors="raise")

    for column in BOOLEAN_COLUMNS:
        remote[column] = remote[column].astype("boolean")

    local = local.set_index("record_id")
    remote = remote.set_index("record_id")

    local_only = local.index.difference(remote.index)
    remote_only = remote.index.difference(local.index)

    print(f"Records only in local CSV: {len(local_only)}")
    print(f"Records only in Hopsworks: {len(remote_only)}")

    if len(local_only) or len(remote_only):
        print("Record sets differ. No data was changed.")
        raise SystemExit(1)

    # Align by identifier, not by row order.
    remote = remote.loc[local.index]

    print("\nMissing-value comparison:")
    missing_counts = pd.DataFrame({
        "local": local.isna().sum(),
        "hopsworks": remote[local.columns].isna().sum(),
    })
    print(missing_counts.to_string())

    print("\nValue comparison:")
    passed = True

    for column in local.columns:
        matches = sum(
            values_match(
                local.at[record_id, column],
                remote.at[record_id, column],
                numeric=column in NUMERIC_COLUMNS,
            )
            for record_id in local.index
        )

        print(f"{column}: {matches}/{len(local)} match")

        if matches != len(local):
            passed = False

    if not passed:
        print("\nVerification failed. Share the output before making changes.")
        raise SystemExit(1)

    print("\nVerification passed.")
    print("Weather values, timestamps, and missing values match.")
    print("No data was uploaded or changed.")


if __name__ == "__main__":
    main()