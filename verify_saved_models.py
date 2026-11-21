import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from train_statistical import load_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

RF_RUN = "random_forest_20261120T135049251958"
SARIMAX_RUN = "sarimax_20261120T141127482438"
LSTM_RUN = "lstm_20261120T175340139816"

VALIDATION_START = "2026-08-01"
VALIDATION_END = "2026-09-30"

RF_METRICS = ["us_aqi", "uv_index"]
LSTM_METRICS = [
    "precipitation_mm",
    "humidity_pct",
    "wind_speed_ms",
]

# Small numerical differences can occur after loading a model.
# Tolerances are fixed before checking the predictions.
TOLERANCES = {
    "random_forest": (1e-8, 1e-8),
    "sarimax": (1e-7, 1e-7),
    "lstm": (1e-5, 1e-5),
}


def target_columns(metric):
    return [
        f"{metric}_t_plus_{horizon}h"
        for horizon in range(1, 25)
    ]


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_saved_predictions(path, metric):
    frame = load_frame(
        path,
        VALIDATION_START,
        VALIDATION_END,
    )
    frame = frame[target_columns(metric)]

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid saved predictions: {path}")

    return frame


def apply_bounds(values, bounds):
    lower, upper = bounds
    return np.clip(
        values,
        -np.inf if lower is None else lower,
        np.inf if upper is None else upper,
    )


def compare_predictions(model_name, metric, predicted, saved):
    if not predicted.index.equals(saved.index):
        raise ValueError(f"{metric}: prediction timestamps differ.")

    if list(predicted.columns) != list(saved.columns):
        raise ValueError(f"{metric}: prediction columns differ.")

    actual_values = predicted.to_numpy(dtype=float)
    saved_values = saved.to_numpy(dtype=float)

    if not np.isfinite(actual_values).all():
        raise ValueError(f"{metric}: non-finite reloaded predictions.")

    atol, rtol = TOLERANCES[model_name]
    matches = np.isclose(
        actual_values,
        saved_values,
        atol=atol,
        rtol=rtol,
    )
    differences = np.abs(actual_values - saved_values)
    passed = bool(matches.all())

    row = {
        "model": model_name,
        "metric": metric,
        "prediction_times": len(saved),
        "values_checked": int(matches.size),
        "values_matching": int(matches.sum()),
        "max_absolute_difference": float(differences.max()),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "passed": passed,
    }

    print(
        f"{metric}: {'PASS' if passed else 'FAIL'} "
        f"— {matches.sum()}/{matches.size} values match; "
        f"maximum difference={differences.max():.10g}",
        flush=True,
    )
    return row


def verify_random_forest():
    print("\nChecking saved Random Forest models...", flush=True)

    run_dir = MODEL_DIR / RF_RUN
    artifact = joblib.load(run_dir / "candidate.joblib")

    features = load_frame(
        DATA_DIR / "supervised" / "validation_X.csv",
        VALIDATION_START,
        VALIDATION_END,
    )

    if list(features.columns) != artifact["feature_columns"]:
        raise ValueError("Random Forest feature names/order differ.")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Invalid Random Forest input features.")

    rows = []

    for metric in RF_METRICS:
        columns = target_columns(metric)

        if artifact["target_columns_by_metric"][metric] != columns:
            raise ValueError(f"Unexpected target order for {metric}.")

        model = artifact["models_by_metric"][metric]
        values = model.predict(features)
        values = apply_bounds(values, artifact["bounds"][metric])

        predicted = pd.DataFrame(
            values,
            index=features.index,
            columns=columns,
        )
        saved = read_saved_predictions(
            run_dir / "validation_predictions.csv", metric
        )

        rows.append(
            compare_predictions(
                "random_forest", metric, predicted, saved
            )
        )

    return rows


def verify_sarimax(train, validation):
    metric = "temperature_c"
    print("\nReconstructing saved temperature SARIMAX...", flush=True)

    run_dir = MODEL_DIR / SARIMAX_RUN / metric
    metadata = read_json(run_dir / "model_specification.json")
    saved = read_saved_predictions(
        run_dir / "validation_predictions.csv", metric
    )

    if metadata["metric"] != metric:
        raise ValueError("Incorrect SARIMAX metric.")

    mean = float(metadata["training_mean"])
    scale = float(metadata["training_scale"])

    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError("Invalid SARIMAX normalization.")

    training_values = train[metric].to_numpy(dtype=float)
    standardized = (training_values - mean) / scale

    model = SARIMAX(
        standardized,
        order=tuple(metadata["order"]),
        seasonal_order=tuple(metadata["seasonal_order"]),
        trend=metadata["trend"],
        enforce_stationarity=metadata["enforce_stationarity"],
        enforce_invertibility=metadata["enforce_invertibility"],
    )

    if list(model.param_names) != metadata["parameter_names"]:
        raise ValueError("SARIMAX parameter names/order differ.")

    parameters = np.asarray(metadata["parameters"], dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError("Invalid saved SARIMAX parameters.")

    # Rebuild the filtered state with fixed parameters.
    # This does not optimize or retrain the model.
    state = model.filter(parameters)

    values = np.empty((len(saved), 24), dtype=float)
    previous_time = train.index[-1]

    for position, prediction_time in enumerate(saved.index):
        if prediction_time != previous_time + pd.Timedelta(hours=1):
            raise ValueError("Gap in SARIMAX replay timestamps.")

        observation = float(validation.at[prediction_time, metric])
        state = state.extend(
            np.array([(observation - mean) / scale], dtype=float)
        )
        values[position] = (
            np.asarray(state.forecast(steps=24), dtype=float)
            .reshape(-1)
            * scale
            + mean
        )
        previous_time = prediction_time

        if (position + 1) % 240 == 0:
            print(
                f"  Replayed {position + 1}/{len(saved)} prediction times",
                flush=True,
            )

    values = apply_bounds(values, metadata["bounds"])
    predicted = pd.DataFrame(
        values,
        index=saved.index,
        columns=target_columns(metric),
    )

    return compare_predictions("sarimax", metric, predicted, saved)


def verify_lstm(train, validation):
    print("\nChecking saved LSTM models...", flush=True)

    history = pd.concat([train, validation])
    expected = pd.date_range(
        "2025-11-01",
        "2026-10-01",
        inclusive="left",
        freq="h",
        tz="UTC",
    )
    if not history.index.equals(expected):
        raise ValueError("LSTM history is not continuous.")

    rows = []

    for metric in LSTM_METRICS:
        run_dir = MODEL_DIR / LSTM_RUN / metric
        metadata = read_json(run_dir / "metadata.json")
        saved = read_saved_predictions(
            run_dir / "validation_predictions.csv", metric
        )
        columns = target_columns(metric)

        if metadata["input_columns"] != [metric]:
            raise ValueError(f"Unexpected LSTM input for {metric}.")
        if metadata["target_columns"] != columns:
            raise ValueError(f"Unexpected LSTM output order for {metric}.")

        mean = float(metadata["training_mean"])
        scale = float(metadata["training_scale"])
        lookback = int(metadata["lookback_hours"])
        batch_size = int(metadata["batch_size"])

        if (
            not np.isfinite([mean, scale]).all()
            or scale <= 0
            or lookback < 1
            or batch_size < 1
        ):
            raise ValueError(f"Invalid LSTM metadata for {metric}.")

        values = (
            (history[metric].to_numpy(dtype=float) - mean) / scale
        ).astype(np.float32)
        positions = history.index.get_indexer(saved.index)

        if (positions < lookback - 1).any():
            raise ValueError(f"Insufficient LSTM history for {metric}.")

        inputs = np.empty(
            (len(saved), lookback, 1), dtype=np.float32
        )
        for row, position in enumerate(positions):
            inputs[row, :, 0] = values[
                position - lookback + 1 : position + 1
            ]

        if not np.isfinite(inputs).all():
            raise ValueError(f"Invalid LSTM inputs for {metric}.")

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            run_dir / "candidate.keras",
            compile=False,
        )

        parts = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            parts.append(model(batch, training=False).numpy())

        predictions = np.concatenate(parts, axis=0) * scale + mean

        if predictions.shape != (len(saved), 24):
            raise ValueError(f"Unexpected LSTM output shape for {metric}.")

        predictions = apply_bounds(predictions, metadata["bounds"])
        predicted = pd.DataFrame(
            predictions,
            index=saved.index,
            columns=columns,
        )

        rows.append(
            compare_predictions("lstm", metric, predicted, saved)
        )

        del model
        tf.keras.backend.clear_session()

    return rows


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("AgentWeather — Saved Model Verification", flush=True)
    print("Checking all 1,440 validation prediction times.")
    print("Six selected outputs; 24 forecast hours per output.")

    train = load_frame(
        DATA_DIR / "splits" / "train.csv",
        "2025-11-01",
        "2026-08-01",
    )
    validation = load_frame(
        DATA_DIR / "splits" / "validation.csv",
        "2026-08-01",
        "2026-10-01",
    )

    required = ["temperature_c", *LSTM_METRICS]
    for name, frame in (("training", train), ("validation", validation)):
        if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid {name} observations.")

    rows = verify_random_forest()
    rows.append(verify_sarimax(train, validation))
    rows.extend(verify_lstm(train, validation))
    results = pd.DataFrame(rows)

    output_dir = DATA_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output = output_dir / f"saved_model_verification_{timestamp}.csv"
    results.to_csv(output, index=False)

    print("\nVerification summary:")
    print(
        results[
            ["model", "metric", "values_matching", "values_checked", "passed"]
        ].to_string(index=False)
    )
    print(f"\nReport saved to: {output}")
    print("No training, model replacement or Hopsworks upload.")
    print("Test files were not read.")
    print("Rain probability and wind direction are not checked by this script.")

    if not results["passed"].all():
        print("\nSome predictions differ. Share the output before proceeding.")
        return 1

    print("\nPASS: all six outputs match within the stated tolerances.")
    print("This verifies saved-model reproduction, not forecast accuracy.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RuntimeError,
        tf.errors.OpError,
    ) as error:
        print(f"\nVerification stopped: {error}")
        raise SystemExit(1) from None