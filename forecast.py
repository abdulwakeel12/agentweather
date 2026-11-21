import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CONTINUOUS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_history(path):
    frame = pd.read_csv(path)
    frame["hour_utc"] = pd.to_datetime(
        frame["hour_utc"], utc=True, errors="raise"
    )
    frame = frame.set_index("hour_utc").sort_index()

    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError(f"Invalid timestamps: {path.name}")

    return frame


def apply_bounds(values, bounds):
    lower, upper = bounds
    return np.clip(
        values,
        -np.inf if lower is None else lower,
        np.inf if upper is None else upper,
    )


class AgentWeatherForecaster:
    """Load saved models once and produce forecasts without training."""

    def __init__(self, project_dir=BASE_DIR):
        self.project_dir = Path(project_dir)
        self.plan = read_json(self.project_dir / "evaluation_plan.json")
        self.artifacts = {}
        self.networks = {}

    def folder(self, metric):
        return self.project_dir / self.plan["selected_models"][metric]["directory"]

    def artifact(self, metric):
        path = self.folder(metric) / "candidate.joblib"
        if path not in self.artifacts:
            self.artifacts[path] = joblib.load(path)
        return self.artifacts[path]

    @staticmethod
    def check_features(features, columns):
        if len(features) != 1:
            raise ValueError("Exactly one feature row is required.")
        if list(features.columns) != list(columns):
            raise ValueError("Feature names or order differ from training.")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Missing or invalid feature values.")

    def forecast(self, origin, features, history):
        """
        origin: UTC prediction timestamp.
        features: one row of the existing 147-feature representation.
        history: hourly observations through origin.

        All outputs are for origin+1h through origin+24h.
        """
        origin = pd.Timestamp(origin)
        if origin.tzinfo is None:
            raise ValueError("Prediction time must include a timezone.")
        origin = origin.tz_convert("UTC")

        if (
            not isinstance(history.index, pd.DatetimeIndex)
            or history.index.tz is None
            or history.index.has_duplicates
            or history.index.hasnans
        ):
            raise ValueError("History requires unique timezone-aware timestamps.")

        history = history.copy()
        history.index = history.index.tz_convert("UTC")
        history = history.sort_index()

        # Reject future observations rather than silently accepting them.
        if history.empty or history.index.max() != origin:
            raise ValueError("History must end exactly at prediction time.")
        if not features.index.equals(pd.DatetimeIndex([origin])):
            raise ValueError("Feature row timestamp must equal prediction time.")

        expected = pd.date_range(history.index.min(), origin, freq="h")
        if not history.index.equals(expected):
            raise ValueError("History contains missing hours.")

        values_by_metric = {}
        model_names = {}

        for metric in CONTINUOUS:
            choice = self.plan["selected_models"][metric]
            model_name = choice["model"]
            model_names[metric] = model_name

            if model_name == "random_forest":
                artifact = self.artifact(metric)
                self.check_features(features, artifact["feature_columns"])
                expected_columns = [
                    f"{metric}_t_plus_{h}h" for h in range(1, 25)
                ]
                if artifact["target_columns_by_metric"][metric] != expected_columns:
                    raise ValueError(f"Unexpected target order: {metric}")

                values = artifact["models_by_metric"][metric].predict(features)[0]
                bounds = artifact["bounds"][metric]

            elif model_name == "lstm":
                import tensorflow as tf

                folder = self.folder(metric)
                metadata = read_json(folder / "metadata.json")
                expected_columns = [
                    f"{metric}_t_plus_{h}h" for h in range(1, 25)
                ]
                if (
                    metadata["input_columns"] != [metric]
                    or metadata["target_columns"] != expected_columns
                ):
                    raise ValueError(f"Unexpected LSTM metadata: {metric}")

                lookback = int(metadata["lookback_hours"])
                mean = float(metadata["training_mean"])
                scale = float(metadata["training_scale"])
                if scale <= 0 or not np.isfinite([mean, scale]).all():
                    raise ValueError("Invalid LSTM normalization.")

                observations = history[metric].tail(lookback).to_numpy(float)
                if len(observations) != lookback or not np.isfinite(observations).all():
                    raise ValueError(f"Insufficient valid history for {metric}")

                inputs = ((observations - mean) / scale).astype(np.float32)
                inputs = inputs.reshape(1, lookback, 1)

                path = folder / "candidate.keras"
                if path not in self.networks:
                    self.networks[path] = tf.keras.models.load_model(
                        path, compile=False
                    )

                values = (
                    self.networks[path](inputs, training=False).numpy()[0]
                    * scale + mean
                )
                bounds = metadata["bounds"]

            elif model_name == "sarimax":
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                metadata = read_json(
                    self.folder(metric) / "model_specification.json"
                )
                if metadata["metric"] != metric:
                    raise ValueError("Incorrect SARIMAX metric.")

                start = pd.Timestamp(metadata["training_start"])
                end = pd.Timestamp(metadata["training_end"])
                if origin < end:
                    raise ValueError("Prediction time precedes SARIMAX training end.")

                training = history.loc[start:end, metric]
                expected_training = pd.date_range(start, end, freq="h")
                if not training.index.equals(expected_training):
                    raise ValueError("SARIMAX requires its complete training history.")

                mean = float(metadata["training_mean"])
                scale = float(metadata["training_scale"])
                if scale <= 0 or not np.isfinite([mean, scale]).all():
                    raise ValueError("Invalid SARIMAX normalization.")

                observations = history.loc[start:origin, metric]
                if not np.isfinite(observations.to_numpy(float)).all():
                    raise ValueError("Invalid SARIMAX observations.")

                model = SARIMAX(
                    (training.to_numpy(float) - mean) / scale,
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
                newer = observations.loc[observations.index > end]
                if len(newer):
                    state = state.extend((newer.to_numpy(float) - mean) / scale)

                values = np.asarray(state.forecast(24)).reshape(-1) * scale + mean
                bounds = metadata["bounds"]

            else:
                raise ValueError(f"Unsupported model: {model_name}")

            values = np.asarray(values, dtype=float)
            if values.shape != (24,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid forecasts for {metric}")
            values_by_metric[metric] = apply_bounds(values, bounds)

        # Individual-hour rain probabilities.
        artifact = self.artifact("rain_probability")
        metadata = artifact["metadata"]
        self.check_features(features, metadata["feature_columns"])
        rain_columns = [
            f"rain_event_t_plus_{h}h" for h in range(1, 25)
        ]
        if metadata["target_columns"] != rain_columns:
            raise ValueError("Rain target order differs.")

        scaler = artifact["scaler"]
        inputs = features if scaler is None else scaler.transform(features)
        probabilities = []

        for column in rain_columns:
            model = artifact["models_by_target"][column]
            if model is None:
                probability = metadata["training_frequency_by_target"][column]
            else:
                positive = np.flatnonzero(np.asarray(model.classes_) == 1)
                if len(positive) != 1:
                    raise ValueError("Missing positive rain class.")
                probability = model.predict_proba(inputs)[0, positive[0]]
            probabilities.append(float(probability))

        probabilities = np.asarray(probabilities)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError("Invalid rain probabilities.")
        values_by_metric["rain_probability"] = probabilities

        # Wind direction is derived from predicted sine and cosine.
        artifact = self.artifact("wind_direction_deg")
        metadata = artifact["metadata"]
        self.check_features(features, metadata["feature_columns"])
        if metadata["component_order"] != ["sin", "cos"]:
            raise ValueError("Unexpected wind component order.")

        epsilon = float(metadata["vector_epsilon"])
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Invalid wind vector threshold.")

        directions = []
        direction_valid = []

        for h in range(1, 25):
            components = artifact["pipelines_by_horizon"][str(h)].predict(features)
            if components.shape != (1, 2) or not np.isfinite(components).all():
                raise ValueError("Invalid predicted wind components.")

            sine, cosine = components[0]
            valid = bool(np.hypot(sine, cosine) > epsilon)
            angle = float(np.rad2deg(np.arctan2(sine, cosine)) % 360)
            directions.append(angle if valid else np.nan)
            direction_valid.append(valid)

        values_by_metric["wind_direction_deg"] = directions

        result = pd.DataFrame({
            "prediction_time_utc": [origin] * 24,
            "horizon_hours": np.arange(1, 25),
            "valid_time_utc": pd.date_range(
                origin + pd.Timedelta(hours=1), periods=24, freq="h"
            ),
            **values_by_metric,
            "wind_direction_vector_valid": direction_valid,
        })
        result["valid_time_pkt"] = result["valid_time_utc"].dt.tz_convert(
            "Asia/Karachi"
        )

        for metric, name in model_names.items():
            result[f"{metric}_model"] = name
        result["rain_probability_model"] = "random_forest"
        result["wind_direction_model"] = "ridge"

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate one historical AgentWeather forecast."
    )
    parser.add_argument(
        "--origin",
        default="2026-09-01T00:00:00+00:00",
        help="Historical prediction timestamp, including timezone.",
    )
    args = parser.parse_args()

    origin = pd.Timestamp(args.origin)
    if origin.tzinfo is None:
        raise ValueError("Use a timezone-aware timestamp.")
    origin = origin.tz_convert("UTC")

    # Keep this initial demonstration within the validation period.
    if not (
        pd.Timestamp("2026-08-01", tz="UTC")
        <= origin
        < pd.Timestamp("2026-09-30", tz="UTC")
    ):
        raise ValueError("This historical demo requires a validation origin.")

    train = read_history(DATA_DIR / "splits/train.csv")
    validation = read_history(DATA_DIR / "splits/validation.csv")
    history = pd.concat([train, validation]).loc[:origin]

    features = read_history(DATA_DIR / "supervised/validation_X.csv")
    features = features.loc[[origin]]

    print(f"\nHistorical prediction time: {origin}", flush=True)
    print("Loading saved models and generating 24 hourly forecasts...", flush=True)

    forecaster = AgentWeatherForecaster()
    forecast = forecaster.forecast(origin, features, history)

    output_dir = DATA_DIR / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output = output_dir / f"historical_forecast_{run_id}.csv"
    forecast.to_csv(output, index=False)

    display_columns = [
        "horizon_hours",
        "valid_time_pkt",
        *CONTINUOUS,
        "rain_probability",
        "wind_direction_deg",
    ]
    print("\nHistorical forecast:")
    print(forecast[display_columns].round(4).to_string(index=False))
    print(f"\nSaved to: {output}")
    print("Rain probability is between 0 and 1 for each individual target hour.")
    print("Historical demonstration, not a current Karachi forecast.")
    print("No rain alerts or live calm-wind display policy applied.")
    print("No retraining, test-file access or Hopsworks upload.")


if __name__ == "__main__":
    main()