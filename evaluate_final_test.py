import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_baseline import regression_scores
from train_statistical import load_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

METRICS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
]

EXPECTED_MODELS = {
    "us_aqi": "random_forest",
    "temperature_c": "sarimax",
    "precipitation_mm": "lstm",
    "humidity_pct": "lstm",
    "uv_index": "random_forest",
    "wind_speed_ms": "lstm",
    "rain_probability": "random_forest",
    "wind_direction_deg": "ridge",
}


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_columns(metric):
    return [f"{metric}_t_plus_{h}h" for h in range(1, 25)]


def check_features(features, expected):
    if list(features.columns) != list(expected):
        raise ValueError("Feature names/order differ from saved model.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite input features.")


def apply_bounds(values, bounds):
    lower, upper = bounds
    return np.clip(
        values,
        -np.inf if lower is None else lower,
        np.inf if upper is None else upper,
    )


def validate_plan(plan):
    if plan["status"] != "frozen_before_test_evaluation":
        raise ValueError("The evaluation plan must be frozen first.")

    if plan["forecast_horizons_hours"] != list(range(1, 25)):
        raise ValueError("Expected horizons 1–24.")

    selections = plan["selected_models"]
    if set(selections) != set(EXPECTED_MODELS):
        raise ValueError("Expected all eight selected outputs.")

    for metric, model in EXPECTED_MODELS.items():
        if selections[metric]["model"] != model:
            raise ValueError(f"Unexpected selected model for {metric}.")

    rules = plan["evaluation_rules"]
    for key in (
        "retrain_before_test",
        "refit_scalers",
        "change_models_using_test_results",
    ):
        if rules[key] is not False:
            raise ValueError(f"Unsupported evaluation rule: {key}")
    if rules["use_saved_bounds"] is not True:
        raise ValueError("Saved bounds must be used.")

    if plan["rain_probability"]["diagnostic_classification_threshold"] != 0.5:
        raise ValueError("This evaluator expects the frozen 0.5 threshold.")

    if plan["rain_probability"]["operational_alert_threshold"] is not None:
        raise ValueError("Operational rain alerts are outside this evaluation.")


def rain_scores(actual, probability, threshold):
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        raise ValueError("Invalid rain probabilities.")

    event = actual.astype(bool)
    predicted = probability >= threshold
    tp = int((event & predicted).sum())
    fp = int((~event & predicted).sum())
    fn = int((event & ~predicted).sum())
    denominator = 2 * tp + fp + fn

    return {
        "brier_score": float(np.mean((probability - actual) ** 2)),
        "actual_rain_hours": int(event.sum()),
        "predicted_rain_hours": int(predicted.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / denominator if denominator else np.nan,
    }


def main():
    plan_path = BASE_DIR / "evaluation_plan.json"
    plan = read_json(plan_path)
    validate_plan(plan)

    # Preflight the required artifacts before loading test files.
    artifact_paths = []
    for metric, choice in plan["selected_models"].items():
        folder = BASE_DIR / choice["directory"]
        if choice["model"] == "lstm":
            names = ["candidate.keras", "metadata.json"]
        elif choice["model"] == "sarimax":
            names = ["model_specification.json"]
        else:
            names = ["candidate.joblib"]
        for name in names:
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing model file: {path}")
            artifact_paths.append(path)

    import tensorflow as tf
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    train = load_frame(
        DATA_DIR / "splits/train.csv", "2025-11-01", "2026-08-01"
    )
    validation = load_frame(
        DATA_DIR / "splits/validation.csv", "2026-08-01", "2026-10-01"
    )
    test = load_frame(
        DATA_DIR / "splits/test.csv", "2026-10-01", "2026-11-01"
    )
    x = load_frame(
        DATA_DIR / "supervised/test_X.csv", "2026-10-01", "2026-10-31"
    )
    y = load_frame(
        DATA_DIR / "supervised/test_y.csv", "2026-10-01", "2026-10-31"
    )

    period = plan["test_period"]
    if (
        len(x) != period["expected_prediction_times"]
        or x.index[0] != pd.Timestamp(period["first_prediction_time_utc"])
        or x.index[-1] != pd.Timestamp(period["last_prediction_time_utc"])
        or x.index[-1] + pd.Timedelta(hours=24)
        != pd.Timestamp(period["last_target_time_utc"])
    ):
        raise ValueError("Test timestamps differ from the frozen plan.")

    history = pd.concat([train, validation, test])
    raw_columns = [*METRICS, "rain_mm", "wind_direction_deg"]
    if not np.isfinite(history[raw_columns].to_numpy(float)).all():
        raise ValueError("Invalid historical observations.")

    # Confirm continuous and rain labels represent the intended future hours.
    # These future values are used only as outcomes for scoring.
    for h in range(1, 25):
        future = test.loc[x.index + pd.Timedelta(hours=h)]
        for metric in METRICS:
            if not np.allclose(
                y[f"{metric}_t_plus_{h}h"].to_numpy(float),
                future[metric].to_numpy(float),
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError(f"Target alignment failed: {metric}, +{h}h")
        if not np.array_equal(
            y[f"rain_event_t_plus_{h}h"].to_numpy(),
            (future["rain_mm"].to_numpy() >= 0.1).astype(int),
        ):
            raise ValueError(f"Rain target alignment failed: +{h}h")

    print("\nAgentWeather — Final October Evaluation", flush=True)
    print(f"Prediction times: {len(x)}")
    print("Forecast horizons: 1–24 hours")
    print("Saved parameters and normalization remain fixed.")

    forecasts = {}
    artifact_cache = {}

    def load_artifact(folder):
        path = folder / "candidate.joblib"
        if path not in artifact_cache:
            artifact_cache[path] = joblib.load(path)
        return artifact_cache[path]

    for metric in METRICS:
        choice = plan["selected_models"][metric]
        folder = BASE_DIR / choice["directory"]
        columns = target_columns(metric)
        print(f"\nPredicting {metric}: {choice['model']}", flush=True)

        if choice["model"] == "random_forest":
            artifact = load_artifact(folder)
            check_features(x, artifact["feature_columns"])
            if artifact["target_columns_by_metric"][metric] != columns:
                raise ValueError("Random Forest output order differs.")
            values = artifact["models_by_metric"][metric].predict(x)
            bounds = artifact["bounds"][metric]

        elif choice["model"] == "lstm":
            metadata = read_json(folder / "metadata.json")
            if (
                metadata["input_columns"] != [metric]
                or metadata["target_columns"] != columns
            ):
                raise ValueError("LSTM column metadata differs.")

            mean = float(metadata["training_mean"])
            scale = float(metadata["training_scale"])
            lookback = int(metadata["lookback_hours"])
            batch = int(metadata["batch_size"])
            if (
                not np.isfinite([mean, scale]).all()
                or scale <= 0 or lookback < 1 or batch < 1
            ):
                raise ValueError("Invalid LSTM normalization/window.")

            standardized = (
                (history[metric].to_numpy(float) - mean) / scale
            ).astype(np.float32)
            positions = history.index.get_indexer(x.index)
            if (positions < lookback - 1).any():
                raise ValueError("Insufficient LSTM history.")

            # Each input ends at its own prediction time, never after it.
            inputs = np.stack([
                standardized[p - lookback + 1:p + 1]
                for p in positions
            ])[:, :, None]

            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(
                folder / "candidate.keras", compile=False
            )
            values = np.concatenate([
                model(inputs[start:start + batch], training=False).numpy()
                for start in range(0, len(inputs), batch)
            ]) * scale + mean
            bounds = metadata["bounds"]
            del model
            tf.keras.backend.clear_session()

        else:
            metadata = read_json(folder / "model_specification.json")
            if metadata["metric"] != metric:
                raise ValueError("SARIMAX metric differs.")

            mean = float(metadata["training_mean"])
            scale = float(metadata["training_scale"])
            if not np.isfinite([mean, scale]).all() or scale <= 0:
                raise ValueError("Invalid SARIMAX normalization.")

            model = SARIMAX(
                (train[metric].to_numpy(float) - mean) / scale,
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

            # Update through all validation observations, without refitting.
            state = state.extend(
                (validation[metric].to_numpy(float) - mean) / scale
            )
            values = np.empty((len(x), 24))
            for position, origin in enumerate(x.index):
                observation = float(test.at[origin, metric])
                state = state.extend(
                    np.array([(observation - mean) / scale])
                )
                values[position] = (
                    np.asarray(state.forecast(24)).reshape(-1) * scale + mean
                )
                if (position + 1) % 240 == 0:
                    print(f"  Forecasted {position + 1}/{len(x)}", flush=True)
            bounds = metadata["bounds"]

        if values.shape != (len(x), 24) or not np.isfinite(values).all():
            raise ValueError(f"Invalid predictions: {metric}")
        forecasts[metric] = pd.DataFrame(
            apply_bounds(values, bounds), index=x.index, columns=columns
        )

    regression_rows = []
    training_precipitation_mean = float(train["precipitation_mm"].mean())

    for metric in METRICS:
        for h, column in enumerate(target_columns(metric), start=1):
            actual = y[column].to_numpy(float)
            previous = history.loc[
                x.index + pd.Timedelta(hours=h - 24), metric
            ].to_numpy(float)
            candidates = {
                plan["selected_models"][metric]["model"]:
                    forecasts[metric][column].to_numpy(),
                "previous_day": previous,
            }
            if metric == "precipitation_mm":
                candidates["zero_precipitation"] = np.zeros(len(x))
                candidates["training_mean_precipitation"] = np.full(
                    len(x), training_precipitation_mean
                )
            for name, predicted in candidates.items():
                regression_rows.append({
                    "metric": metric,
                    "model": name,
                    "horizon_hours": h,
                    **regression_scores(actual, predicted),
                })

    print("\nPredicting rain probabilities...", flush=True)
    folder = BASE_DIR / plan["selected_models"]["rain_probability"]["directory"]
    artifact = load_artifact(folder)
    metadata = artifact["metadata"]
    check_features(x, metadata["feature_columns"])
    rain_columns = target_columns("rain_event")
    if metadata["target_columns"] != rain_columns:
        raise ValueError("Rain target order differs.")

    inputs = x if artifact["scaler"] is None else artifact["scaler"].transform(x)
    rain_predictions = {}
    rain_rows = []
    reliability = []
    threshold = plan["rain_probability"]["diagnostic_classification_threshold"]

    for h, column in enumerate(rain_columns, start=1):
        model = artifact["models_by_target"][column]
        frequency = float(metadata["training_frequency_by_target"][column])
        if not np.isfinite(frequency) or not 0 <= frequency <= 1:
            raise ValueError("Invalid saved training rain frequency.")

        if model is None:
            probability = np.full(len(x), frequency)
        else:
            positive = np.flatnonzero(np.asarray(model.classes_) == 1)
            if len(positive) != 1:
                raise ValueError("Missing positive rain class.")
            probability = model.predict_proba(inputs)[:, positive[0]]

        rain_predictions[column] = probability
        actual = y[column].to_numpy(float)
        previous = (
            history.loc[
                x.index + pd.Timedelta(hours=h - 24), "rain_mm"
            ].to_numpy() >= 0.1
        ).astype(float)

        candidates = {
            "random_forest": probability,
            "always_dry": np.zeros(len(x)),
            "training_frequency": np.full(len(x), frequency),
            "previous_day": previous,
        }
        for name, probabilities in candidates.items():
            rain_rows.append({
                "model": name,
                "horizon_hours": h,
                **rain_scores(actual, probabilities, threshold),
            })
            bins = np.minimum((probabilities * 10).astype(int), 9)
            for number in range(10):
                mask = bins == number
                count = int(mask.sum())
                reliability.append({
                    "model": name,
                    "horizon_hours": h,
                    "bin_lower": number / 10,
                    "bin_upper": (number + 1) / 10,
                    "count": count,
                    "mean_probability": (
                        float(probabilities[mask].mean()) if count else np.nan
                    ),
                    "observed_rain_fraction": (
                        float(actual[mask].mean()) if count else np.nan
                    ),
                })

    print("\nPredicting wind direction...", flush=True)
    folder = BASE_DIR / plan["selected_models"]["wind_direction_deg"]["directory"]
    artifact = load_artifact(folder)
    metadata = artifact["metadata"]
    check_features(x, metadata["feature_columns"])
    if metadata["component_order"] != ["sin", "cos"]:
        raise ValueError("Wind component order differs.")

    epsilon = float(metadata["vector_epsilon"])
    calm = float(metadata["calm_threshold_ms"])
    if (
        epsilon != plan["wind_direction"]["vector_epsilon"]
        or calm != plan["wind_direction"]["calm_threshold_ms"]
    ):
        raise ValueError("Wind evaluation rules differ from saved metadata.")

    direction_predictions = {}
    direction_rows = []

    for h in range(1, 25):
        components = artifact["pipelines_by_horizon"][str(h)].predict(x)
        if components.shape != (len(x), 2) or not np.isfinite(components).all():
            raise ValueError("Invalid wind components.")

        magnitude = np.hypot(components[:, 0], components[:, 1])
        valid_prediction = magnitude > epsilon
        predicted = np.rad2deg(
            np.arctan2(components[:, 0], components[:, 1])
        ) % 360
        predicted[~valid_prediction] = np.nan
        direction_predictions[f"wind_direction_deg_t_plus_{h}h"] = predicted

        future = test.loc[x.index + pd.Timedelta(hours=h)]
        previous = history.loc[x.index + pd.Timedelta(hours=h - 24)]
        target_valid = future["wind_speed_ms"].to_numpy() > calm
        baseline_valid = previous["wind_speed_ms"].to_numpy() > calm
        common = target_valid & baseline_valid & valid_prediction
        actual = future["wind_direction_deg"].to_numpy(float)

        counts = {
            "horizon_hours": h,
            "evaluated_rows": int(common.sum()),
            "excluded_target_calm": int((~target_valid).sum()),
            "excluded_baseline_calm_after_target_filter": int(
                (target_valid & ~baseline_valid).sum()
            ),
            "excluded_zero_vector_after_calm_filters": int(
                (target_valid & baseline_valid & ~valid_prediction).sum()
            ),
        }
        for name, candidate in (
            ("ridge", predicted),
            ("previous_day", previous["wind_direction_deg"].to_numpy(float)),
        ):
            errors = (candidate[common] - actual[common] + 180) % 360 - 180
            direction_rows.append({
                "model": name,
                **counts,
                "angular_mae_deg": (
                    float(np.abs(errors).mean()) if len(errors) else np.nan
                ),
                "angular_rmse_deg": (
                    float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan
                ),
            })

    regression = pd.DataFrame(regression_rows)
    rain = pd.DataFrame(rain_rows)
    direction = pd.DataFrame(direction_rows)

    regression_summary = regression.groupby(
        ["metric", "model"], as_index=False
    ).agg(mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"))
    rain_summary = rain.groupby("model", as_index=False).agg(
        mean_brier_score=("brier_score", "mean")
    )
    direction_summary = direction.groupby("model", as_index=False).agg(
        mean_angular_mae_deg=("angular_mae_deg", "mean"),
        mean_angular_rmse_deg=("angular_rmse_deg", "mean"),
        horizons_evaluated=("angular_mae_deg", "count"),
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output = DATA_DIR / "evaluation" / f"final_test_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    tables = {
        "regression_by_horizon": regression,
        "regression_summary": regression_summary,
        "rain_by_horizon": rain,
        "rain_summary": rain_summary,
        "rain_reliability": pd.DataFrame(reliability),
        "wind_direction_by_horizon": direction,
        "wind_direction_summary": direction_summary,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    pd.concat(forecasts.values(), axis=1).to_csv(
        output / "continuous_predictions.csv", index_label="hour_utc"
    )
    pd.DataFrame(rain_predictions, index=x.index).to_csv(
        output / "rain_probabilities.csv", index_label="hour_utc"
    )
    pd.DataFrame(direction_predictions, index=x.index).to_csv(
        output / "wind_direction_predictions.csv", index_label="hour_utc"
    )
    with (output / "evaluation_plan.json").open("w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2)

    fingerprint_paths = set(artifact_paths + [
        plan_path,
        Path(__file__).resolve(),
        DATA_DIR / "splits/train.csv",
        DATA_DIR / "splits/validation.csv",
        DATA_DIR / "splits/test.csv",
        DATA_DIR / "supervised/test_X.csv",
        DATA_DIR / "supervised/test_y.csv",
    ])
    hashes = {
        str(path.relative_to(BASE_DIR)): fingerprint(path)
        for path in sorted(fingerprint_paths)
    }
    with (output / "file_hashes.json").open("w", encoding="utf-8") as file:
        json.dump(hashes, file, indent=2)

    print("\nMean test errors across 24 horizons:")
    print(regression_summary.round(6).to_string(index=False))

    print("\nRegression results at +24 hours:")
    print(
        regression.loc[regression["horizon_hours"].eq(24)]
        .round(6).to_string(index=False)
    )

    print("\nRain probability — mean Brier score:")
    print(rain_summary.round(6).to_string(index=False))

    print("\nRain diagnostics at +24 hours — threshold 0.5:")
    print(
        rain.loc[rain["horizon_hours"].eq(24)]
        .round(6).to_string(index=False)
    )

    print("\nWind-direction test results:")
    print(direction_summary.round(6).to_string(index=False))

    print(f"\nResults saved to: {output}")
    print("October test data was evaluated.")
    print("No model parameters, scalers or thresholds were fitted.")
    print("No models replaced. No Hopsworks upload.")
    print("Report these results honestly; do not tune against this test month.")


if __name__ == "__main__":
    main()