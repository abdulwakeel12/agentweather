import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
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
    path = Path(path)
    frame = pd.read_csv(path)
    frame["hour_utc"] = pd.to_datetime(
        frame["hour_utc"], utc=True, errors="raise"
    )
    frame = frame.set_index("hour_utc").sort_index()

    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError(f"Invalid timestamps: {path.name}")

    return frame


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def apply_bounds(values, bounds):
    lower, upper = bounds
    return np.clip(
        values,
        -np.inf if lower is None else lower,
        np.inf if upper is None else upper,
    )


class AgentWeatherForecaster:
    """Load saved models and generate predictions without training."""

    def __init__(self, project_dir=BASE_DIR):
        self.project_dir = Path(project_dir)
        self.plan = read_json(self.project_dir / "evaluation_plan.json")
        self.artifacts = {}
        self.networks = {}

    def folder(self, metric):
        choice = self.plan["selected_models"][metric]
        return self.project_dir / choice["directory"]

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
        Generate forecasts for origin +1h through origin +24h.

        features: one row of the existing 147-feature representation.
        history: continuous hourly history ending exactly at origin.
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
            raise ValueError(
                "History requires unique timezone-aware timestamps."
            )

        history = history.copy()
        history.index = history.index.tz_convert("UTC")
        history = history.sort_index()

        if history.empty or history.index.max() != origin:
            raise ValueError("History must end exactly at prediction time.")

        if not history.index.equals(history.index.floor("h")):
            raise ValueError("History timestamps must fall on exact hours.")

        if not features.index.equals(pd.DatetimeIndex([origin])):
            raise ValueError(
                "Feature row timestamp must equal prediction time."
            )

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
                actual_columns = artifact["target_columns_by_metric"][metric]

                if actual_columns != expected_columns:
                    raise ValueError(f"Unexpected target order: {metric}")

                values = artifact["models_by_metric"][metric].predict(
                    features
                )[0]
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

                if (
                    len(observations) != lookback
                    or not np.isfinite(observations).all()
                ):
                    raise ValueError(
                        f"Insufficient valid history for {metric}"
                    )

                inputs = ((observations - mean) / scale).astype(np.float32)
                inputs = inputs.reshape(1, lookback, 1)

                path = folder / "candidate.keras"
                if path not in self.networks:
                    self.networks[path] = tf.keras.models.load_model(
                        path, compile=False
                    )

                values = (
                    self.networks[path](inputs, training=False).numpy()[0]
                    * scale
                    + mean
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
                    raise ValueError(
                        "Prediction time precedes SARIMAX training end."
                    )

                training = history.loc[start:end, metric]
                expected_training = pd.date_range(start, end, freq="h")

                if not training.index.equals(expected_training):
                    raise ValueError(
                        "SARIMAX requires its complete training history."
                    )

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

                parameters = np.asarray(
                    metadata["parameters"], dtype=float
                )
                if not np.isfinite(parameters).all():
                    raise ValueError("Invalid SARIMAX parameters.")

                state = model.filter(parameters)
                newer = observations.loc[observations.index > end]

                if len(newer):
                    state = state.extend(
                        (newer.to_numpy(float) - mean) / scale
                    )

                values = (
                    np.asarray(state.forecast(24)).reshape(-1) * scale
                    + mean
                )
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

        # Wind direction from predicted sine and cosine.
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

        for horizon in range(1, 25):
            components = artifact["pipelines_by_horizon"][
                str(horizon)
            ].predict(features)

            if components.shape != (1, 2) or not np.isfinite(components).all():
                raise ValueError("Invalid predicted wind components.")

            sine, cosine = components[0]
            valid = bool(np.hypot(sine, cosine) > epsilon)
            angle = float(np.rad2deg(np.arctan2(sine, cosine)) % 360)

            directions.append(angle if valid else np.nan)
            direction_valid.append(valid)

        values_by_metric["wind_direction_deg"] = directions

        result = pd.DataFrame(
            {
                "prediction_time_utc": [origin] * 24,
                "horizon_hours": np.arange(1, 25),
                "valid_time_utc": pd.date_range(
                    origin + pd.Timedelta(hours=1),
                    periods=24,
                    freq="h",
                ),
                **values_by_metric,
                "wind_direction_vector_valid": direction_valid,
            }
        )

        result["valid_time_pkt"] = result["valid_time_utc"].dt.tz_convert(
            "Asia/Karachi"
        )

        for metric, name in model_names.items():
            result[f"{metric}_model"] = name

        result["rain_probability_model"] = "random_forest"
        result["wind_direction_model"] = "ridge"

        return result


def experimental_forecast(snapshot_path, history_path):
    """Run a separately labelled experiment using recent GFS inputs."""
    from build_features import METRICS, make_features

    snapshot_path = Path(snapshot_path).resolve()
    history_path = Path(history_path).resolve()

    paths = [
        snapshot_path,
        history_path,
        BASE_DIR / "evaluation_plan.json",
    ]
    hashes = {str(path): file_hash(path) for path in paths}

    snapshot = read_history(snapshot_path)
    historical = read_history(history_path)
    required = [*METRICS, "wind_direction_deg"]
    now = pd.Timestamp.now(tz="UTC")

    if len(snapshot) != 169:
        raise ValueError(
            "Experimental mode requires exactly 169 snapshot rows."
        )

    expected = pd.date_range(
        snapshot.index.min(), periods=169, freq="h"
    )
    if not snapshot.index.equals(expected):
        raise ValueError("Snapshot hours must be consecutive.")

    if not snapshot.index.equals(snapshot.index.floor("h")):
        raise ValueError("Snapshot timestamps must fall on exact hours.")

    origin = snapshot.index.max()
    transition = snapshot.index.min()

    if origin > now:
        raise ValueError("Snapshot contains future valid times.")

    for column, source in (
        ("weather_source", "open_meteo_gfs_global"),
        ("air_source", "open_meteo_cams_global"),
    ):
        if not snapshot[column].eq(source).all():
            raise ValueError(f"Unexpected snapshot source: {column}")

    retrievals = {}

    for column in (
        "weather_retrieved_at_utc",
        "air_retrieved_at_utc",
    ):
        times = pd.to_datetime(
            snapshot[column], utc=True, errors="raise"
        )

        if times.isna().any() or times.gt(now).any():
            raise ValueError(f"Invalid retrieval timestamps: {column}")

        retrievals[column] = times.max().isoformat()

    # Keep ERA5 before the transition and GFS from the transition onward.
    # Overlapping source rows are replaced, not averaged.
    older = historical.loc[historical.index < transition].copy()

    if (
        older.empty
        or older.index.max() + pd.Timedelta(hours=1) != transition
    ):
        raise ValueError(
            "Historical data does not connect to the snapshot."
        )

    if not older["weather_source"].eq("open_meteo_era5").all():
        raise ValueError(
            "Expected ERA5 weather before the source transition."
        )

    if not older["air_source"].eq("open_meteo_cams_global").all():
        raise ValueError("Expected CAMS air quality in historical data.")

    history = pd.concat([older[required], snapshot[required]])
    history = history.apply(pd.to_numeric, errors="raise")

    # Reuse the exact existing feature calculations.
    # make_features also checks continuity and finite input values.
    features = make_features(history).loc[[origin]]

    if features.shape != (1, 147):
        raise ValueError("Expected one row containing 147 features.")

    forecaster = AgentWeatherForecaster(project_dir=BASE_DIR)

    temperature_choice = forecaster.plan["selected_models"]["temperature_c"]
    if temperature_choice["model"] != "sarimax":
        raise ValueError(
            "This experimental setup expects SARIMAX for temperature."
        )

    specification_path = (
        forecaster.folder("temperature_c") / "model_specification.json"
    )
    hashes[str(specification_path)] = file_hash(specification_path)
    specification = read_json(specification_path)

    training_start = pd.Timestamp(specification["training_start"])
    training_end = pd.Timestamp(specification["training_end"])

    if transition <= training_end:
        raise ValueError(
            "Source transition must be after SARIMAX training."
        )

    # Confirm the SARIMAX training temperatures have not been replaced.
    saved_training_path = DATA_DIR / "splits/train.csv"
    hashes[str(saved_training_path)] = file_hash(saved_training_path)
    saved_training = read_history(saved_training_path)

    expected_training = pd.date_range(
        training_start, training_end, freq="h"
    )
    reference = saved_training.loc[
        training_start:training_end, "temperature_c"
    ]
    actual = history.loc[
        training_start:training_end, "temperature_c"
    ]

    if (
        not actual.index.equals(expected_training)
        or not reference.index.equals(expected_training)
        or not np.isfinite(reference.to_numpy(float)).all()
        or not np.allclose(
            actual.to_numpy(float),
            reference.to_numpy(float),
            atol=1e-10,
            rtol=0,
        )
    ):
        raise ValueError(
            "SARIMAX training history differs from saved training data."
        )

    input_age = (now - origin).total_seconds() / 3600

    print(
        "AgentWeather — Experimental Recent-Input Forecast",
        flush=True,
    )
    print("Input cutoff:", origin)
    print("Input age in hours:", round(input_age, 2))
    print("Weather source transition:", transition)
    print("Historical ERA5 rows:", len(older))
    print("Recent GFS rows:", len(snapshot))
    print("Feature shape:", features.shape)
    print("SARIMAX training-history check: PASS")
    print(
        "Accuracy with GFS inputs is not established.",
        flush=True,
    )

    result = forecaster.forecast(origin, features, history)
    generated_at = datetime.now(UTC)
    mode = "experimental_mixed_source"

    result["generated_at_utc"] = generated_at.isoformat()
    result["forecast_mode"] = mode
    result["input_cutoff_utc"] = origin.isoformat()
    result["weather_source_transition_utc"] = transition.isoformat()
    result["recent_weather_source"] = "open_meteo_gfs_global"
    result["historical_weather_source"] = "open_meteo_era5"

    for column, value in retrievals.items():
        result[column] = value

    result["target_time_before_generation"] = (
        result["valid_time_utc"] < pd.Timestamp(generated_at)
    )

    for filename, digest in hashes.items():
        if file_hash(filename) != digest:
            raise ValueError(
                f"Input file changed during execution: {filename}"
            )

    run_id = generated_at.strftime("%Y%m%dT%H%M%S%f")
    output_dir = DATA_DIR / "experimental_forecasts" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    result.to_csv(output_dir / "forecast.csv", index=False)
    features.to_csv(
        output_dir / "features.csv",
        index_label="hour_utc",
    )

    manifest = {
        "mode": mode,
        "generated_at_utc": generated_at.isoformat(),
        "input_cutoff_utc": origin.isoformat(),
        "weather_source_transition_utc": transition.isoformat(),
        "retrieval_times": retrievals,
        "input_file_sha256": hashes,
        "selected_models": forecaster.plan["selected_models"],
        "historical_rows": len(older),
        "snapshot_rows": len(snapshot),
        "targets_before_generation": int(
            result["target_time_before_generation"].sum()
        ),
        "limitations": [
            "GFS-input forecast accuracy has not been established.",
            (
                "SARIMAX state uses ERA5 followed by GFS "
                "with fixed parameters."
            ),
            (
                "Input cutoff is not the forecast issuance "
                "or retrieval time."
            ),
            (
                "Targets before generation time are "
                "retrospective predictions."
            ),
            "Rain probabilities are not established as calibrated.",
            "No alerts, notifications or dashboard integration.",
        ],
    }

    with (output_dir / "manifest.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            allow_nan=False,
        )

    display_columns = [
        "horizon_hours",
        "valid_time_pkt",
        *CONTINUOUS,
        "rain_probability",
        "wind_direction_deg",
    ]

    print("\nExperimental predictions:")
    print(result[display_columns].round(4).to_string(index=False))
    print("\nSaved to:", output_dir)
    print(
        "Target times already passed when generated:",
        int(result["target_time_before_generation"].sum()),
    )
    print("Experimental output only; not a validated live forecast.")
    print("No retraining, dashboard changes, alerts or Hopsworks upload.")


def historical_forecast(origin_text):
    origin = pd.Timestamp(origin_text)

    if origin.tzinfo is None:
        raise ValueError("Use a timezone-aware timestamp.")

    origin = origin.tz_convert("UTC")

    if not (
        pd.Timestamp("2026-08-01", tz="UTC")
        <= origin
        < pd.Timestamp("2026-09-30", tz="UTC")
    ):
        raise ValueError(
            "This historical demo requires a validation origin."
        )

    train = read_history(DATA_DIR / "splits/train.csv")
    validation = read_history(DATA_DIR / "splits/validation.csv")
    history = pd.concat([train, validation]).loc[:origin]

    features = read_history(
        DATA_DIR / "supervised/validation_X.csv"
    ).loc[[origin]]

    print(f"\nHistorical prediction time: {origin}", flush=True)
    print(
        "Loading saved models and generating 24 hourly forecasts...",
        flush=True,
    )

    forecaster = AgentWeatherForecaster()
    result = forecaster.forecast(origin, features, history)

    generated_at = datetime.now(UTC)
    result["generated_at_utc"] = generated_at.isoformat()
    result["forecast_mode"] = "retrospective_historical_demo"

    output_dir = DATA_DIR / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = generated_at.strftime("%Y%m%dT%H%M%S%f")
    output = output_dir / f"historical_forecast_{run_id}.csv"
    result.to_csv(output, index=False)

    display_columns = [
        "horizon_hours",
        "valid_time_pkt",
        *CONTINUOUS,
        "rain_probability",
        "wind_direction_deg",
    ]

    print("\nHistorical forecast:")
    print(result[display_columns].round(4).to_string(index=False))
    print(f"\nSaved to: {output}")
    print(
        "Rain probability is between 0 and 1 "
        "for each individual target hour."
    )
    print("Historical demonstration, not a current Karachi forecast.")
    print("No rain alerts or live calm-wind display policy applied.")
    print("No retraining, test-file access or Hopsworks upload.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a historical or experimental AgentWeather forecast."
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "experimental"],
        default="historical",
    )
    parser.add_argument(
        "--origin",
        default="2026-09-01T00:00:00+00:00",
        help="Historical prediction timestamp, including timezone.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=(
            DATA_DIR
            / "recent_inputs/20261122T104510468620_e198e792"
            / "recent_hourly_inputs.csv"
        ),
        help="Saved recent-input snapshot for experimental mode.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=(
            DATA_DIR
            / "inference/history_through_2026-11-15.csv"
        ),
        help="Historical data used to reconstruct SARIMAX state.",
    )

    args = parser.parse_args()

    if args.mode == "experimental":
        experimental_forecast(args.snapshot, args.history)
    else:
        historical_forecast(args.origin)


if __name__ == "__main__":
    main()