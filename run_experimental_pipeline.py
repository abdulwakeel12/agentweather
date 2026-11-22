import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

import collect_recent_inputs as collector
from build_features import METRICS, make_features
from forecast import (
    AgentWeatherForecaster,
    build_experimental_baselines,
    file_hash,
    read_history,
    read_json,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HISTORICAL_PATH = (
    DATA_DIR / "inference/history_through_2026-11-15.csv"
)
SEED_SNAPSHOT = (
    DATA_DIR
    / "recent_inputs/20261122T104510468620_e198e792"
    / "recent_hourly_inputs.csv"
)

STATE_DIR = DATA_DIR / "experimental_state"
CHECKPOINT = STATE_DIR / "history.csv"
LOCK_PATH = STATE_DIR / "pipeline.lock"

REQUIRED = [*METRICS, "wind_direction_deg"]


def save_json(path, content):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)


def validate_history(frame):
    if frame.empty or frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError("History has empty or invalid timestamps.")

    if not frame.index.equals(frame.index.floor("h")):
        raise ValueError("History timestamps must be exact hours.")

    expected = pd.date_range(
        frame.index.min(), frame.index.max(), freq="h"
    )
    if not frame.index.equals(expected):
        missing = expected.difference(frame.index)
        raise ValueError(
            f"History contains gaps. First missing hours: {list(missing[:5])}"
        )

    frame = frame.copy()
    frame[REQUIRED] = frame[REQUIRED].apply(
        pd.to_numeric, errors="raise"
    )

    if not np.isfinite(frame[REQUIRED].to_numpy(float)).all():
        raise ValueError("History contains missing or non-finite values.")

    collector.validate_combined(frame)

    if not frame["air_source"].eq("open_meteo_cams_global").all():
        raise ValueError("Unexpected air-quality source.")

    weather_sources = frame["weather_source"]
    allowed = {"open_meteo_era5", "open_meteo_gfs_global"}

    if not weather_sources.isin(allowed).all():
        raise ValueError("Unexpected weather source.")

    gfs_rows = weather_sources.eq("open_meteo_gfs_global")
    if not gfs_rows.any():
        raise ValueError("Expected experimental GFS history.")

    transition = frame.index[gfs_rows][0]

    if not weather_sources.loc[
        weather_sources.index < transition
    ].eq("open_meteo_era5").all():
        raise ValueError("Expected ERA5 before the source transition.")

    if not weather_sources.loc[transition:].eq(
        "open_meteo_gfs_global"
    ).all():
        raise ValueError("Weather sources switch back after GFS begins.")

    return frame, transition


def load_previous_history():
    if CHECKPOINT.exists():
        return validate_history(read_history(CHECKPOINT))

    historical = read_history(HISTORICAL_PATH)
    seed = read_history(SEED_SNAPSHOT)

    if len(seed) != 169:
        raise ValueError("Expected 169 rows in the initial GFS snapshot.")

    if not seed["weather_source"].eq("open_meteo_gfs_global").all():
        raise ValueError("The initial snapshot must contain GFS weather.")

    transition = seed.index.min()
    older = historical.loc[historical.index < transition]

    combined = pd.concat([older, seed], sort=False)
    return validate_history(combined)


def check_training_history(history, forecaster, transition):
    choice = forecaster.plan["selected_models"]["temperature_c"]
    if choice["model"] != "sarimax":
        raise ValueError("This workflow expects temperature SARIMAX.")

    specification = read_json(
        forecaster.folder("temperature_c") / "model_specification.json"
    )

    start = pd.Timestamp(specification["training_start"])
    end = pd.Timestamp(specification["training_end"])

    if transition <= end:
        raise ValueError("GFS must not replace SARIMAX training history.")

    saved = read_history(DATA_DIR / "splits/train.csv")
    reference = saved.loc[start:end, "temperature_c"]
    actual = history.loc[start:end, "temperature_c"]
    expected = pd.date_range(start, end, freq="h")

    if (
        not reference.index.equals(expected)
        or not actual.index.equals(expected)
        or not np.isfinite(reference.to_numpy(float)).all()
        or not np.allclose(
            actual.to_numpy(float),
            reference.to_numpy(float),
            atol=1e-10,
            rtol=0,
        )
    ):
        raise ValueError("SARIMAX training-history verification failed.")


def run_pipeline():
    started = datetime.now(UTC)
    run_id = started.strftime("%Y%m%dT%H%M%S%f") + "_" + uuid4().hex[:8]
    run_dir = DATA_DIR / "experimental_forecasts" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log = {
        "run_id": run_id,
        "mode": "experimental_mixed_source",
        "started_at_utc": started.isoformat(),
        "status": "running",
    }

    try:
        print("AgentWeather — Experimental Collection and Forecast Pipeline")
        print("Run:", run_id)

        print("\nStep 1/4: Read continuous history", flush=True)
        previous, transition = load_previous_history()

        print("\nStep 2/4: Collect a fresh snapshot", flush=True)

        # Give this collector invocation a private output directory.
        # Its standalone behavior and source file remain unchanged.
        original_base = collector.BASE_DIR
        collector.BASE_DIR = run_dir

        try:
            collection_status = collector.main()
        finally:
            collector.BASE_DIR = original_base

        if collection_status != 0:
            raise ValueError("Input collection failed; forecasting stopped.")

        snapshot_paths = list(
            (run_dir / "data/recent_inputs").glob(
                "*/recent_hourly_inputs.csv"
            )
        )
        if len(snapshot_paths) != 1:
            raise ValueError("Expected exactly one newly collected snapshot.")

        snapshot_path = snapshot_paths[0]
        snapshot = read_history(snapshot_path)
        manifest = read_json(snapshot_path.parent / "manifest.json")

        if manifest["status"] != "validated" or len(snapshot) != 169:
            raise ValueError("The new snapshot is not validated.")

        if not snapshot["weather_source"].eq(
            "open_meteo_gfs_global"
        ).all():
            raise ValueError("Unexpected snapshot weather source.")

        now = pd.Timestamp.now(tz="UTC")
        for column in (
            "weather_retrieved_at_utc",
            "air_retrieved_at_utc",
        ):
            times = pd.to_datetime(
                snapshot[column], utc=True, errors="raise"
            )
            if times.isna().any() or times.gt(now).any():
                raise ValueError(f"Invalid retrieval time: {column}")

        origin = snapshot.index.max()

        if origin > now:
            raise ValueError("Input cutoff is in the future.")

        if origin < previous.index.max():
            raise ValueError("New snapshot is older than saved history.")

        # Retain older inputs; replace overlapping recent hours with this
        # run's snapshot. Never fill gaps or fabricate observations.
        recent = snapshot.loc[snapshot.index >= transition]
        if recent.empty:
            raise ValueError("Snapshot does not reach the GFS history.")

        older = previous.loc[previous.index < recent.index.min()]
        history = pd.concat([older, recent], sort=False)
        history, actual_transition = validate_history(history)

        if actual_transition != transition:
            raise ValueError("The original weather-source transition changed.")

        if history.index.max() != origin:
            raise ValueError("Combined history does not end at the new cutoff.")

        features = make_features(history).loc[[origin]]
        if features.shape != (1, 147):
            raise ValueError("Expected exactly 147 input features.")

        print("\nStep 3/4: Generate predictions and baselines", flush=True)
        print("Input cutoff:", origin)
        print("Source transition:", transition)
        print("Continuous history rows:", len(history))

        forecaster = AgentWeatherForecaster(project_dir=BASE_DIR)
        check_training_history(history, forecaster, transition)
        print("SARIMAX training-history check: PASS", flush=True)

        result = forecaster.forecast(origin, features, history)
        generated = datetime.now(UTC)

        result["generated_at_utc"] = generated.isoformat()
        result["forecast_mode"] = "experimental_mixed_source"
        result["input_cutoff_utc"] = origin.isoformat()
        result["weather_source_transition_utc"] = transition.isoformat()
        result["recent_weather_source"] = "open_meteo_gfs_global"
        result["historical_weather_source"] = "open_meteo_era5"
        result["target_time_before_generation"] = (
            result["valid_time_utc"] < pd.Timestamp(generated)
        )

        for column in (
            "weather_retrieved_at_utc",
            "air_retrieved_at_utc",
        ):
            result[column] = snapshot[column].iloc[-1]

        baselines = build_experimental_baselines(snapshot, result)

        print("\nStep 4/4: Save results and history checkpoint", flush=True)
        history_path = run_dir / "input_history.csv"
        history.to_csv(history_path, index_label="hour_utc")
        features.to_csv(run_dir / "features.csv", index_label="hour_utc")
        result.to_csv(run_dir / "forecast.csv", index=False)
        baselines.to_csv(run_dir / "baseline_predictions.csv", index=False)

        log.update(
            {
                "generated_at_utc": generated.isoformat(),
                "input_cutoff_utc": origin.isoformat(),
                "source_transition_utc": transition.isoformat(),
                "snapshot": str(snapshot_path.relative_to(run_dir)),
                "history_rows": len(history),
                "forecast_rows": len(result),
                "baseline_rows": len(baselines),
                "input_history_sha256": file_hash(history_path),
                "snapshot_sha256": file_hash(snapshot_path),
                "evaluation_plan_sha256": file_hash(
                    BASE_DIR / "evaluation_plan.json"
                ),
                "selected_models": forecaster.plan["selected_models"],
                "accuracy_with_recent_inputs_established": False,
            }
        )
        save_json(run_dir / "manifest.json", log)

        # Advance the checkpoint only after all forecast artifacts exist.
        temporary = STATE_DIR / f"history_{run_id}.tmp"
        history.to_csv(temporary, index_label="hour_utc")
        temporary.replace(CHECKPOINT)

        log["status"] = "success"
        log["completed_at_utc"] = datetime.now(UTC).isoformat()
        save_json(run_dir / "execution_log.json", log)

        print("\nExperimental pipeline completed.")
        print("Forecast rows:", len(result))
        print("Baseline rows:", len(baselines))
        print("Saved to:", run_dir)
        print("History checkpoint:", CHECKPOINT)
        print("Accuracy with GFS inputs remains unestablished.")
        print("No dashboard changes, alerts, retraining or Hopsworks upload.")
        print("\nEvaluate this specific run later with:")
        print(f'python evaluate_experimental.py --run "{run_dir}"')

        return 0

    except Exception as error:  # noqa: BLE001
        # Record failures at the workflow boundary and return a failing exit code.
        log["status"] = "failed"
        log["error"] = f"{type(error).__name__}: {error}"
        log["completed_at_utc"] = datetime.now(UTC).isoformat()

        try:
            save_json(run_dir / "failure.json", log)
        except OSError:
            print("Could not write the failure log.")

        print(f"\nExperimental pipeline failed: {error}")
        print("Run folder:", run_dir)
        return 1


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        lock = LOCK_PATH.open("x", encoding="utf-8")
    except FileExistsError:
        print("Another run may be active.")
        print("Lock file:", LOCK_PATH)
        print("If a previous run was interrupted, confirm it has stopped")
        print("before removing that lock file.")
        return 1

    try:
        with lock:
            lock.write(datetime.now(UTC).isoformat())
        return run_pipeline()
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())