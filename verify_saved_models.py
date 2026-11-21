import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_statistical import load_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

RF_RUN = "random_forest_20261120T135049251958"
SARIMAX_RUN = "sarimax_20261120T141127482438"
LSTM_RUN = "lstm_20261120T175340139816"
RAIN_RUN = "rain_random_forest_20261120T190019747084"
WIND_RUN = "wind_direction_ridge_20261120T191558239804"

START = "2026-08-01"
END = "2026-09-30"


def columns_for(metric):
    return [f"{metric}_t_plus_{h}h" for h in range(1, 25)]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_validation(path):
    return load_frame(path, START, END)


def check_features(features, expected):
    if list(features.columns) != list(expected):
        raise ValueError("Saved feature names/order differ from validation.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite validation features.")


def bounded(values, bounds):
    lower, upper = bounds
    return np.clip(
        values,
        -np.inf if lower is None else lower,
        np.inf if upper is None else upper,
    )


def compare(model, metric, predicted, saved, atol, rtol=0, angular=False):
    if not predicted.index.equals(saved.index):
        raise ValueError(f"{metric}: timestamps differ.")
    if list(predicted.columns) != list(saved.columns):
        raise ValueError(f"{metric}: columns differ.")

    new = predicted.to_numpy(dtype=float)
    old = saved.to_numpy(dtype=float)

    if np.isinf(new).any() or np.isinf(old).any():
        raise ValueError(f"{metric}: infinite predictions.")

    if not angular and (np.isnan(new).any() or np.isnan(old).any()):
        raise ValueError(f"{metric}: unexpected missing predictions.")

    finite = np.isfinite(new) & np.isfinite(old)
    both_missing = np.isnan(new) & np.isnan(old)
    differences = np.full(new.shape, np.nan)

    if angular:
        for values in (new, old):
            valid = values[np.isfinite(values)]
            if ((valid < 0) | (valid > 360)).any():
                raise ValueError("Wind direction outside 0–360 degrees.")
        differences[finite] = np.abs(
            (new[finite] - old[finite] + 180) % 360 - 180
        )
    else:
        differences[finite] = np.abs(new[finite] - old[finite])

    matches = both_missing.copy()
    matches[finite] = (
        differences[finite] <= atol + rtol * np.abs(old[finite])
    )
    passed = bool(matches.all() and finite.any())
    maximum = float(differences[finite].max()) if finite.any() else np.nan

    row = {
        "model": model,
        "metric": metric,
        "prediction_times": len(saved),
        "values_checked": int(matches.size),
        "values_matching": int(matches.sum()),
        "both_undefined": int(both_missing.sum()),
        "max_absolute_difference": maximum,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "passed": passed,
    }
    print(
        f"{metric}: {'PASS' if passed else 'FAIL'} — "
        f"{matches.sum()}/{matches.size} match; "
        f"maximum difference={maximum:.10g}",
        flush=True,
    )
    return row


def verify_rain(features):
    print("\nChecking saved rain-probability models...", flush=True)
    folder = MODEL_DIR / RAIN_RUN
    artifact = joblib.load(folder / "candidate.joblib")
    metadata = artifact["metadata"]
    columns = columns_for("rain_event")

    check_features(features, metadata["feature_columns"])
    if metadata["model"] != "random_forest":
        raise ValueError("Unexpected rain model.")
    if metadata["target_columns"] != columns:
        raise ValueError("Unexpected rain target order.")
    if set(artifact["models_by_target"]) != set(columns):
        raise ValueError("Missing or unexpected rain models.")

    scaler = artifact["scaler"]
    inputs = features if scaler is None else scaler.transform(features)
    parts = {}

    for column in columns:
        model = artifact["models_by_target"][column]
        if model is None:
            frequency = metadata["training_frequency_by_target"][column]
            probabilities = np.full(len(features), float(frequency))
        else:
            positive = np.flatnonzero(np.asarray(model.classes_) == 1)
            if len(positive) != 1:
                raise ValueError(f"Missing positive rain class: {column}")
            probabilities = model.predict_proba(inputs)[:, positive[0]]

        parts[column] = probabilities

    predicted = pd.DataFrame(parts, index=features.index)
    saved = read_validation(folder / "validation_probabilities.csv")

    if list(saved.columns) != columns:
        raise ValueError("Saved rain probability columns differ.")

    for frame in (predicted, saved):
        values = frame.to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
        ):
            raise ValueError("Invalid rain probabilities.")

    return compare(
        "random_forest", "rain_probability",
        predicted, saved, atol=1e-10, rtol=1e-8,
    )


def verify_wind(features):
    print("\nChecking saved wind-direction models...", flush=True)
    folder = MODEL_DIR / WIND_RUN
    artifact = joblib.load(folder / "candidate.joblib")
    metadata = artifact["metadata"]

    check_features(features, metadata["feature_columns"])
    if metadata["component_order"] != ["sin", "cos"]:
        raise ValueError("Unexpected wind component order.")
    if set(artifact["pipelines_by_horizon"]) != {
        str(h) for h in range(1, 25)
    }:
        raise ValueError("Missing or unexpected wind models.")

    epsilon = float(metadata["vector_epsilon"])
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid wind vector threshold.")

    directions = {}
    components = {}

    for horizon in range(1, 25):
        model = artifact["pipelines_by_horizon"][str(horizon)]
        values = model.predict(features)

        if (
            values.shape != (len(features), 2)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid wind output at +{horizon}h.")

        sine, cosine = values[:, 0], values[:, 1]
        magnitude = np.hypot(sine, cosine)
        angle = np.rad2deg(np.arctan2(sine, cosine)) % 360
        angle[magnitude <= epsilon] = np.nan

        directions[f"wind_direction_deg_t_plus_{horizon}h"] = angle
        components[f"predicted_sin_t_plus_{horizon}h"] = sine
        components[f"predicted_cos_t_plus_{horizon}h"] = cosine
        components[f"vector_magnitude_t_plus_{horizon}h"] = magnitude

    predicted = pd.DataFrame(directions, index=features.index)
    saved = read_validation(folder / "validation_predictions.csv")

    # Also verify the components used to calculate the direction.
    predicted_components = pd.DataFrame(components, index=features.index)
    diagnostics = read_validation(folder / "validation_diagnostics.csv")
    saved_components = diagnostics[list(predicted_components.columns)]

    component_result = compare(
        "ridge", "wind_direction_components",
        predicted_components, saved_components,
        atol=1e-8, rtol=1e-8,
    )
    direction_result = compare(
        "ridge", "wind_direction_deg",
        predicted, saved, atol=1e-5, angular=True,
    )
    return [component_result, direction_result]


def verify_continuous(features):
    import tensorflow as tf
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    rows = []

    print("\nChecking continuous Random Forest models...", flush=True)
    folder = MODEL_DIR / RF_RUN
    artifact = joblib.load(folder / "candidate.joblib")
    check_features(features, artifact["feature_columns"])
    saved_rf = read_validation(folder / "validation_predictions.csv")

    for metric in ("us_aqi", "uv_index"):
        columns = columns_for(metric)
        if artifact["target_columns_by_metric"][metric] != columns:
            raise ValueError(f"Unexpected target order: {metric}")

        values = artifact["models_by_metric"][metric].predict(features)
        predicted = pd.DataFrame(
            bounded(values, artifact["bounds"][metric]),
            index=features.index,
            columns=columns,
        )
        rows.append(compare(
            "random_forest", metric, predicted, saved_rf[columns],
            atol=1e-8, rtol=1e-8,
        ))

    train = load_frame(
        DATA_DIR / "splits" / "train.csv",
        "2025-11-01", "2026-08-01",
    )
    validation = load_frame(
        DATA_DIR / "splits" / "validation.csv",
        "2026-08-01", "2026-10-01",
    )
    metrics = [
        "temperature_c", "precipitation_mm",
        "humidity_pct", "wind_speed_ms",
    ]
    for frame in (train, validation):
        if not np.isfinite(frame[metrics].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite historical observations.")

    print("\nReconstructing temperature SARIMAX...", flush=True)
    metric = "temperature_c"
    folder = MODEL_DIR / SARIMAX_RUN / metric
    metadata = read_json(folder / "model_specification.json")
    saved = read_validation(folder / "validation_predictions.csv")
    columns = columns_for(metric)

    if metadata["metric"] != metric or list(saved.columns) != columns:
        raise ValueError("Unexpected SARIMAX metadata or columns.")

    mean = float(metadata["training_mean"])
    scale = float(metadata["training_scale"])
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError("Invalid SARIMAX normalization.")

    model = SARIMAX(
        (train[metric].to_numpy(dtype=float) - mean) / scale,
        order=tuple(metadata["order"]),
        seasonal_order=tuple(metadata["seasonal_order"]),
        trend=metadata["trend"],
        enforce_stationarity=metadata["enforce_stationarity"],
        enforce_invertibility=metadata["enforce_invertibility"],
    )
    if list(model.param_names) != metadata["parameter_names"]:
        raise ValueError("SARIMAX parameter order differs.")

    parameters = np.asarray(metadata["parameters"], dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError("Invalid SARIMAX parameters.")

    state = model.filter(parameters)
    values = np.empty((len(saved), 24))
    previous_time = train.index[-1]

    for position, origin in enumerate(saved.index):
        if origin != previous_time + pd.Timedelta(hours=1):
            raise ValueError("Gap in SARIMAX replay.")
        observed = float(validation.at[origin, metric])
        state = state.extend(np.array([(observed - mean) / scale]))
        values[position] = (
            np.asarray(state.forecast(24)).reshape(-1) * scale + mean
        )
        previous_time = origin
        if (position + 1) % 240 == 0:
            print(f"  Replayed {position + 1}/{len(saved)}", flush=True)

    predicted = pd.DataFrame(
        bounded(values, metadata["bounds"]),
        index=saved.index, columns=columns,
    )
    rows.append(compare(
        "sarimax", metric, predicted, saved, atol=1e-7, rtol=1e-7
    ))

    history = pd.concat([train, validation])
    print("\nChecking saved LSTM models...", flush=True)

    for metric in ("precipitation_mm", "humidity_pct", "wind_speed_ms"):
        folder = MODEL_DIR / LSTM_RUN / metric
        metadata = read_json(folder / "metadata.json")
        saved = read_validation(folder / "validation_predictions.csv")
        columns = columns_for(metric)

        if (
            metadata["input_columns"] != [metric]
            or metadata["target_columns"] != columns
            or list(saved.columns) != columns
        ):
            raise ValueError(f"Unexpected LSTM columns: {metric}")

        mean = float(metadata["training_mean"])
        scale = float(metadata["training_scale"])
        lookback = int(metadata["lookback_hours"])
        batch_size = int(metadata["batch_size"])
        if (
            not np.isfinite([mean, scale]).all()
            or scale <= 0 or lookback < 1 or batch_size < 1
        ):
            raise ValueError(f"Invalid LSTM metadata: {metric}")

        standardized = (
            (history[metric].to_numpy(dtype=float) - mean) / scale
        ).astype(np.float32)
        positions = history.index.get_indexer(saved.index)
        if (positions < lookback - 1).any():
            raise ValueError("Insufficient LSTM history.")

        inputs = np.stack([
            standardized[p - lookback + 1:p + 1]
            for p in positions
        ])[:, :, None]

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            folder / "candidate.keras", compile=False
        )
        parts = [
            model(inputs[start:start + batch_size], training=False).numpy()
            for start in range(0, len(inputs), batch_size)
        ]
        values = np.concatenate(parts) * scale + mean
        if values.shape != (len(saved), 24):
            raise ValueError("Unexpected LSTM output shape.")

        predicted = pd.DataFrame(
            bounded(values, metadata["bounds"]),
            index=saved.index, columns=columns,
        )
        rows.append(compare(
            "lstm", metric, predicted, saved, atol=1e-5, rtol=1e-5
        ))
        del model
        tf.keras.backend.clear_session()

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Verify saved AgentWeather predictions without retraining."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also repeat verification of the six continuous outputs.",
    )
    args = parser.parse_args()

    print("AgentWeather — Saved Model Verification")
    print("Scope:", "all eight forecast outputs" if args.all else
          "rain probability and wind direction")

    features = read_validation(
        DATA_DIR / "supervised" / "validation_X.csv"
    )

    rows = [verify_rain(features)]
    rows.extend(verify_wind(features))
    if args.all:
        rows.extend(verify_continuous(features))

    results = pd.DataFrame(rows)
    output_dir = DATA_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output = output_dir / f"saved_model_verification_{timestamp}.csv"
    results.to_csv(output, index=False)

    print("\nVerification summary:")
    print(results[
        ["metric", "values_matching", "values_checked", "passed"]
    ].to_string(index=False))

    print(f"\nReport saved to: {output}")
    print("No models trained or replaced.")
    print("Test files not read. No Hopsworks upload.")
    print("No rain thresholds changed or calibration performed.")

    if not results["passed"].all():
        print("\nFAIL: share the output before proceeding.")
        return 1

    print("\nPASS: all requested checks matched within tolerance.")
    print("This verifies reproduction, not forecast accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())