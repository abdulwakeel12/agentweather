import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from evaluate_baseline import regression_scores
from train_statistical import load_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

METRIC = "temperature_c"
LOOKBACK = 72
HORIZONS = 24
BATCH_SIZE = 32
MAX_EPOCHS = 40
SEED = 42

TARGET_COLUMNS = [
    f"{METRIC}_t_plus_{horizon}h"
    for horizon in range(1, HORIZONS + 1)
]


def load_data():
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

    train_targets = load_frame(
        DATA_DIR / "supervised" / "train_y.csv",
        "2025-11-08",
        "2026-07-31",
    )[TARGET_COLUMNS]

    validation_targets = load_frame(
        DATA_DIR / "supervised" / "validation_y.csv",
        "2026-08-01",
        "2026-09-30",
    )[TARGET_COLUMNS]

    history = pd.concat(
        [train[[METRIC]], validation[[METRIC]]]
    )

    expected = pd.date_range(
        "2025-11-01",
        "2026-10-01",
        inclusive="left",
        freq="h",
        tz="UTC",
        name="hour_utc",
    )

    if not history.index.equals(expected):
        raise ValueError("Historical timestamps are not continuous.")

    for name, frame in (
        ("history", history),
        ("training targets", train_targets),
        ("validation targets", validation_targets),
    ):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or invalid values in {name}.")

    # Verify target values and their exact future timestamps.
    for targets in (train_targets, validation_targets):
        for horizon, column in enumerate(TARGET_COLUMNS, start=1):
            future_times = targets.index + pd.Timedelta(hours=horizon)
            expected_values = history.loc[
                future_times, METRIC
            ].to_numpy(dtype=float)

            if not np.allclose(
                targets[column].to_numpy(dtype=float),
                expected_values,
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError(f"Target alignment failed: {column}.")

    if (
        train_targets.index.max() + pd.Timedelta(hours=HORIZONS)
        >= validation_targets.index.min()
    ):
        raise ValueError("Training targets overlap validation.")

    return train, history, train_targets, validation_targets


def make_sequences(history, origins, mean, scale):
    values = (
        (history[METRIC].to_numpy(dtype=float) - mean) / scale
    ).astype(np.float32)

    positions = history.index.get_indexer(origins)

    if (positions < LOOKBACK - 1).any():
        raise ValueError("An origin has insufficient historical context.")

    sequences = np.empty(
        (len(origins), LOOKBACK, 1),
        dtype=np.float32,
    )

    for row, position in enumerate(positions):
        # Inclusive endpoint: observations t-71 through t.
        # No observations later than t enter this input.
        sequences[row, :, 0] = values[
            position - LOOKBACK + 1 : position + 1
        ]

    if not np.isfinite(sequences).all():
        raise ValueError("Invalid sequence values.")

    return sequences


def make_dataset(inputs, targets):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(BATCH_SIZE)

    # Limit background work on the local computer.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1

    return dataset.with_options(options).prefetch(1)


def load_baseline():
    baseline = pd.read_csv(
        DATA_DIR / "evaluation" / "baseline_regression_by_horizon.csv"
    )

    baseline = baseline.loc[
        baseline["metric"].eq(METRIC),
        ["metric", "horizon_hours", "mae", "rmse", "r2"],
    ].copy()

    if (
        len(baseline) != HORIZONS
        or baseline["horizon_hours"].duplicated().any()
        or set(baseline["horizon_hours"])
        != set(range(1, HORIZONS + 1))
    ):
        raise ValueError("Baseline results are incomplete or duplicated.")

    rmse = baseline["rmse"].to_numpy(dtype=float)

    if not np.isfinite(rmse).all() or (rmse < 0).any():
        raise ValueError("Invalid baseline RMSE.")

    return baseline


def main():
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.utils.set_random_seed(SEED)

        train, history, train_targets, validation_targets = load_data()
        baseline = load_baseline()

        # Scaling uses training observations only.
        mean = float(train[METRIC].mean())
        scale = float(train[METRIC].std(ddof=0))

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid training scale.")

        x_train = make_sequences(
            history, train_targets.index, mean, scale
        )
        x_validation = make_sequences(
            history, validation_targets.index, mean, scale
        )

        y_train = (
            (train_targets.to_numpy(dtype=float) - mean) / scale
        ).astype(np.float32)

        y_validation = (
            (validation_targets.to_numpy(dtype=float) - mean) / scale
        ).astype(np.float32)

        print(f"\nMetric: {METRIC}")
        print(f"Training examples: {len(x_train)}")
        print(f"Validation examples: {len(x_validation)}")
        print(f"Input shape: {x_train.shape}")
        print(f"Lookback: {LOOKBACK} hours")
        print(f"Forecast horizons: 1–{HORIZONS} hours")
        print("Scaling fitted on training observations only.")

        train_dataset = make_dataset(x_train, y_train)
        validation_dataset = make_dataset(
            x_validation, y_validation
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(LOOKBACK, 1)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(HORIZONS),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0,
            ),
            loss="mse",
        )

        model.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )

        training_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=MAX_EPOCHS,
            callbacks=[
                early_stopping,
                tf.keras.callbacks.TerminateOnNaN(),
            ],
            verbose=2,
        )

        losses = np.asarray(training_history.history["loss"])
        validation_losses = np.asarray(
            training_history.history["val_loss"]
        )

        if (
            not np.isfinite(losses).all()
            or not np.isfinite(validation_losses).all()
        ):
            raise ValueError("Training produced non-finite losses.")

        # Predict in small batches without creating another data pipeline.
        parts = []

        for start in range(0, len(x_validation), BATCH_SIZE):
            batch = x_validation[start : start + BATCH_SIZE]
            parts.append(model(batch, training=False).numpy())

        predictions = np.concatenate(parts, axis=0) * scale + mean

        if predictions.shape != validation_targets.shape:
            raise ValueError("Unexpected prediction shape.")

        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite predictions.")

        prediction_frame = pd.DataFrame(
            predictions,
            index=validation_targets.index,
            columns=TARGET_COLUMNS,
        )

        score_rows = []

        for horizon, column in enumerate(TARGET_COLUMNS, start=1):
            score_rows.append(
                {
                    "metric": METRIC,
                    "horizon_hours": horizon,
                    **regression_scores(
                        validation_targets[column].to_numpy(dtype=float),
                        prediction_frame[column].to_numpy(dtype=float),
                    ),
                }
            )

        scores = pd.DataFrame(score_rows)

        comparison = scores.merge(
            baseline,
            on=["metric", "horizon_hours"],
            validate="one_to_one",
            suffixes=("_lstm", "_baseline"),
        )

        mean_lstm_rmse = float(comparison["rmse_lstm"].mean())
        mean_baseline_rmse = float(comparison["rmse_baseline"].mean())
        best_epoch = int(np.argmin(validation_losses)) + 1

        print("\nTemperature results at +24 hours:")
        print(
            comparison.loc[
                comparison["horizon_hours"].eq(24),
                [
                    "metric",
                    "horizon_hours",
                    "mae_lstm",
                    "rmse_lstm",
                    "rmse_baseline",
                    "r2_lstm",
                ],
            ].round(4).to_string(index=False)
        )

        print("\nMean validation RMSE across 24 horizons:")
        print(f"LSTM:     {mean_lstm_rmse:.4f}")
        print(f"Baseline: {mean_baseline_rmse:.4f}")
        print(f"LSTM better: {mean_lstm_rmse < mean_baseline_rmse}")
        print(f"Best epoch: {best_epoch}")

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_dir = DATA_DIR / "models" / f"lstm_temperature_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)

        model.save(run_dir / "candidate.keras")

        prediction_frame.to_csv(
            run_dir / "validation_predictions.csv",
            index_label="hour_utc",
        )
        comparison.to_csv(
            run_dir / "baseline_comparison.csv",
            index=False,
        )

        history_frame = pd.DataFrame(training_history.history)
        history_frame.index = np.arange(1, len(history_frame) + 1)
        history_frame.to_csv(
            run_dir / "training_history.csv",
            index_label="epoch",
        )

        metadata = {
            "metric": METRIC,
            "input_columns": [METRIC],
            "lookback_hours": LOOKBACK,
            "input_window": "t-71 through t inclusive",
            "target_columns": TARGET_COLUMNS,
            "training_mean": mean,
            "training_scale": scale,
            "scaling_period_start": train.index.min().isoformat(),
            "scaling_period_end": train.index.max().isoformat(),
            "training_origin_start": train_targets.index.min().isoformat(),
            "training_origin_end": train_targets.index.max().isoformat(),
            "validation_origin_start": (
                validation_targets.index.min().isoformat()
            ),
            "validation_origin_end": (
                validation_targets.index.max().isoformat()
            ),
            "lstm_units": 32,
            "batch_size": BATCH_SIZE,
            "epochs_completed": len(history_frame),
            "best_epoch": best_epoch,
            "seed": SEED,
            "tensorflow_version": tf.__version__,
            "mean_validation_rmse": mean_lstm_rmse,
            "mean_baseline_rmse": mean_baseline_rmse,
            "selection": "Best epoch by validation scaled MSE",
            "availability_assumption": (
                "Observation at time t is available at time t; "
                "actual source publication delays are not simulated."
            ),
            "test_data_used": False,
        }

        with (run_dir / "metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2, allow_nan=False)

        print(f"\nSaved to: {run_dir}")
        print("Validation was used for early stopping.")
        print("Results are provisional, not final test performance.")
        print("No existing model replaced.")
        print("Test data not read. No Hopsworks upload.")
        return 0

    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"\nLSTM training failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())