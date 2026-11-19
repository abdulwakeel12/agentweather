import hashlib
import os
from pathlib import Path

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from prepare_data import prepare_source

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def make_record_id(row):
    identity = (
        f"{row['source']}|"
        f"{row['grid_latitude']:.6f}|"
        f"{row['grid_longitude']:.6f}|"
        f"{row['valid_time_utc'].isoformat()}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main():
    local = prepare_source("air_quality")
    local["record_id"] = local.apply(make_record_id, axis=1)

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
        name="karachi_air_quality_estimates",
        version=1,
    )

    print("\nReading saved records...")
    remote = fg.read()

    print(f"Local records: {len(local)}")
    print(f"Hopsworks records: {len(remote)}")

    columns = ["valid_time_utc", "retrieved_at_utc"]

    print("\nReturned timestamp types:")
    print(remote[columns].dtypes.to_string())

    print("\nReturned values before conversion:")
    print(remote[["record_id", *columns]].to_string(index=False))

    # Do not guess the unit if the client returns numeric timestamps.
    for column in columns:
        if pd.api.types.is_numeric_dtype(remote[column]):
            print(f"\n{column} was returned as a number.")
            print("Stop here and share the output so we can verify its unit.")
            raise SystemExit(1)

        remote[column] = pd.to_datetime(
            remote[column],
            utc=True,
            errors="raise",
        )

    if remote["record_id"].duplicated().any():
        print("\nDuplicate record IDs found in Hopsworks.")
        raise SystemExit(1)

    comparison = local[["record_id", *columns]].merge(
        remote[["record_id", *columns]],
        on="record_id",
        how="outer",
        suffixes=("_local", "_hopsworks"),
        indicator=True,
        validate="one_to_one",
    )

    matched = comparison["_merge"].eq("both")
    passed = matched.all()

    print("\nTimestamp comparison:")
    for column in columns:
        same = (
            matched
            & comparison[f"{column}_local"].eq(
                comparison[f"{column}_hopsworks"]
            )
        )
        print(f"{column}: {same.sum()}/{len(comparison)} match")
        passed = passed and same.all()

    if not passed:
        print("\nDifferences or records present in only one location:")
        print(comparison.to_string(index=False))
        print("\nNo data was changed. Share this output before proceeding.")
        raise SystemExit(1)

    print("\nVerification passed: both timestamps match the local CSV.")
    print("No data was uploaded or changed.")


if __name__ == "__main__":
    main()