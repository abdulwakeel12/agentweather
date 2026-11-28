"""Validate one cloud run and prepare its saved outputs for deployment."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

METRICS = (
    "temperature_c", "humidity_pct", "us_aqi", "uv_index",
    "wind_speed_ms", "precipitation_mm", "rain_probability",
)


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timezone missing from timestamp.")
    return result.astimezone(UTC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("data/experimental_forecasts"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    logs = list(args.runs.glob("*/execution_log.json"))
    if len(logs) != 1:
        raise ValueError("Expected exactly one run in this clean cloud workspace.")
    log_path = logs[0]
    run_dir = log_path.parent
    log = json.loads(log_path.read_text(encoding="utf-8"))
    run_id = log["run_id"]
    if not re.fullmatch(r"\d{8}T\d{12}_[a-f0-9]{8}", run_id):
        raise ValueError("Invalid run ID.")
    if run_id != run_dir.name or log.get("status") != "success":
        raise ValueError("Run is not a matching successful execution.")
    if log.get("mode") != "experimental_mixed_source":
        raise ValueError("Expected experimental output.")
    origin = timestamp(log["input_cutoff_utc"])
    generated = timestamp(log["generated_at_utc"])
    if not origin <= generated <= datetime.now(UTC):
        raise ValueError("Invalid cutoff or generation time.")

    files = {}
    tables = {}
    for name, count in (("forecast.csv", 24), ("baseline_predictions.csv", 48)):
        content = (run_dir / name).read_bytes()
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        if len(rows) != count:
            raise ValueError(f"Unexpected row count in {name}.")
        files[name] = content
        tables[name] = rows
    if log["forecast_rows"] != 24 or log["baseline_rows"] != 48:
        raise ValueError("Execution log row counts differ.")

    forecast = tables["forecast.csv"]
    if sorted(int(row["horizon_hours"]) for row in forecast) != list(range(1, 25)):
        raise ValueError("Forecast horizons must be 1 through 24.")
    for row in forecast:
        horizon = int(row["horizon_hours"])
        if (
            timestamp(row["prediction_time_utc"]) != origin
            or timestamp(row["generated_at_utc"]) != generated
            or timestamp(row["valid_time_utc"]) != origin + timedelta(hours=horizon)
            or row["forecast_mode"] != "experimental_mixed_source"
        ):
            raise ValueError("Forecast metadata mismatch.")
        values = {metric: float(row[metric]) for metric in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Non-finite forecast values.")
        if not 0 <= values["humidity_pct"] <= 100:
            raise ValueError("Invalid humidity.")
        if not 0 <= values["rain_probability"] <= 1:
            raise ValueError("Invalid probability.")
        if any(values[key] < 0 for key in (
            "us_aqi", "uv_index", "wind_speed_ms", "precipitation_mm"
        )):
            raise ValueError("Negative forecast metric.")

    baselines = tables["baseline_predictions.csv"]
    expected = {(method, horizon) for method in ("persistence", "previous_day")
                for horizon in range(1, 25)}
    actual = {(row["baseline"], int(row["horizon_hours"])) for row in baselines}
    if actual != expected:
        raise ValueError("Baseline methods or horizons differ.")
    for row in baselines:
        if (
            timestamp(row["prediction_time_utc"]) != origin
            or timestamp(row["forecast_generated_at_utc"]) != generated
            or timestamp(row["valid_time_utc"]) != origin
            + timedelta(hours=int(row["horizon_hours"]))
        ):
            raise ValueError("Baseline timestamp mismatch.")

    history_hash = hashlib.sha256((run_dir / "input_history.csv").read_bytes()).hexdigest()
    if history_hash != log["input_history_sha256"]:
        raise ValueError("Input history checksum mismatch.")

    destination = args.destination.resolve()
    if destination.name != "cloud_data" or destination.parent.name != "deployment":
        raise ValueError("Destination must be deployment/cloud_data.")
    existing = destination / "publication.json"
    if destination.exists() and not existing.is_file():
        raise ValueError("Refusing to replace an unmanaged directory.")
    if existing.is_file():
        previous = json.loads(existing.read_text())
        if timestamp(previous["generated_at_utc"]) >= generated:
            raise ValueError("Refusing to replace an equal or newer publication.")

    files["execution_log.json"] = log_path.read_bytes()
    publication = {
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "input_cutoff_utc": origin.isoformat(),
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    if destination.exists():
        shutil.rmtree(destination)
    output = destination / "experimental_forecasts" / run_id
    output.mkdir(parents=True)
    for name, content in files.items():
        (output / name).write_bytes(content)
    (destination / "publication.json").write_text(
        json.dumps(publication, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared verified dashboard publication:", run_id)
    print("Forecast rows: 24; baseline rows: 48")
    print("Only forecast, baselines, execution log and publication metadata included.")


if __name__ == "__main__":
    main()
