import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from evaluate_baseline import regression_scores
from train_statistical import BOUNDS, METRICS, load_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LOOKBACK = 72
HORIZONS = 24
BATCH_SIZE = 32
MAX_EPOCHS = 40
SEED = 42


def target_columns(metric):
    return [
        f"{metric}_t_plus_{horizon}h"
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
    )
    validation_targets = load_frame(
        DATA_DIR / "supervised" / "validation_y.csv",
        "2026-08-01",
        "2026-09-30",
    )

    history = pd.concat([train[METRICS], validation[METRICS]])

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

    if (
        train_targets.index.max() + pd.Timedelta(hours=HORIZONS)
        >= validation_targets.index.min()
    ):
        raise ValueError("Training targets overlap validation.")

    baseline = pd.read_csv(
        DATA_DIR / "evaluation" / "baseline_regression_by_horizon.csv"
    )

    return train, history, train_targets, validation_targets, baseline


def validate_metric(metric, history, train_y, val_y, baseline):
    columns = target_columns(metric)

    for frame in (
        history[[metric]],
        train_y[columns],
        val_y[columns],
    ):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or invalid values for {metric}.")

    for targets in (train_y, val_y):
        for horizon, column in enumerate(columns, start=1):
            future_times = targets.index + pd.Timedelta(hours=horizon)
            expected = history.loc[
                future_times, metric
            ].to_numpy(dtype=float)

            if not np.allclose(
                targets[column].to_numpy(dtype=float),
                expected,
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError(f"Target alignment failed: {column}.")

    reference = baseline.loc[
        baseline["metric"].eq(metric),
        ["metric", "horizon_hours", "mae", "rmse", "r2"],
    ].copy()

    if (
        len(reference) != HORIZONS
        or reference["horizon_hours"].duplicated().any()
        or set(reference["horizon_hours"])
        != set(range(1, HORIZONS + 1))
    ):
        raise ValueError(f"Incomplete baseline for {metric}.")

    rmse = reference["rmse"].to_numpy(dtype=float)
    if not np.isfinite(rmse).all() or (rmse < 0).any():
        raise ValueError(f"Invalid baseline RMSE for {metric}.")

    return columns, reference


def make_sequences(history, metric, origins, mean, scale):
    values = (
        (history[metric].to_numpy(dtype=float) - mean) / scale
    ).astype(np.float32)

    positions = history.index.get_indexer(origins)

    if (positions < LOOKBACK - 1).any():
        raise ValueError("Missing origin or insufficient history.")

    sequences = np.empty(
        (len(origins), LOOKBACK, 1),
        dtype=np.float32,
    )

    for row, position in enumerate(positions):
        # Include t-71 through t, never observations after t.
        sequences[row, :, 0] = values[
            position - LOOKBACK + 1 : position + 1
        ]

    if not np.isfinite(sequences).all():
        raise ValueError("Invalid sequence values.")

    return sequences


def make_dataset(inputs, targets):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(BATCH_SIZE)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1

    return dataset.with_options(options).prefetch(1)


def score_predictions(metric, actual, predictions):
    rows = []

    for horizon, column in enumerate(target_columns(metric), start=1):
        rows.append(
            {
                "metric": metric,
                "horizon_hours": horizon,
                **regression_scores(
                    actual[column].to_numpy(dtype=float),
                    predictions[column].to_numpy(dtype=float),
                ),
            }
        )

    return pd.DataFrame(rows)


def train_metric(metric, data, run_dir):
    train, history, train_y, val_y, baseline = data

    columns, reference = validate_metric(
        metric, history, train_y, val_y, baseline
    )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    # Fit normalization on training observations only.
    mean = float(train[metric].mean())
    scale = float(train[metric].std(ddof=0))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid training scale for {metric}.")

    x_train = make_sequences(
        history, metric, train_y.index, mean, scale
    )
    x_val = make_sequences(
        history, metric, val_y.index, mean, scale
    )

    y_train = (
        (train_y[columns].to_numpy(dtype=float) - mean) / scale
    ).astype(np.float32)
    y_val = (
        (val_y[columns].to_numpy(dtype=float) - mean) / scale
    ).astype(np.float32)

    print(f"\n{'=' * 55}")
    print(f"Training LSTM: {metric}", flush=True)
    print(f"Training examples: {len(x_train)}")
    print(f"Validation examples: {len(x_val)}")
    print(f"Input shape: {x_train.shape}")

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

    training_history = model.fit(
        make_dataset(x_train, y_train),
        validation_data=make_dataset(x_val, y_val),
        epochs=MAX_EPOCHS,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    )

    history_frame = pd.DataFrame(training_history.history)
    if not np.isfinite(
        history_frame[["loss", "val_loss"]].to_numpy()
    ).all():
        raise ValueError(f"Non-finite training losses for {metric}.")

    best_epoch = int(
        np.argmin(history_frame["val_loss"].to_numpy())
    ) + 1

    parts = []
    for start in range(0, len(x_val), BATCH_SIZE):
        batch = x_val[start : start + BATCH_SIZE]
        parts.append(model(batch, training=False).numpy())

    values = np.concatenate(parts, axis=0) * scale + mean

    if (
        values.shape != y_val.shape
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid predictions for {metric}.")

    raw = pd.DataFrame(
        values,
        index=val_y.index,
        columns=columns,
    )

    lower, upper = BOUNDS[metric]
    bounded = raw.clip(lower=lower, upper=upper)

    raw_scores = score_predictions(metric, val_y, raw)
    bounded_scores = score_predictions(metric, val_y, bounded)

    comparison = bounded_scores.merge(
        reference,
        on=["metric", "horizon_hours"],
        validate="one_to_one",
        suffixes=("_lstm", "_baseline"),
    )

    lstm_rmse = float(comparison["rmse_lstm"].mean())
    baseline_rmse = float(comparison["rmse_baseline"].mean())
    changed_values = int(
        (raw.to_numpy() != bounded.to_numpy()).sum()
    )

    summary = {
        "metric": metric,
        "lstm_mean_rmse": lstm_rmse,
        "baseline_mean_rmse": baseline_rmse,
        "lstm_better": lstm_rmse < baseline_rmse,
        "best_epoch": best_epoch,
        "epochs_completed": len(history_frame),
        "predictions_clipped": changed_values,
    }

    metric_dir = run_dir / metric
    metric_dir.mkdir(parents=True, exist_ok=False)

    # The saved network produces standardized, unbounded predictions.
    # Inference must invert normalization and then apply the bounds.
    model.save(metric_dir / "candidate.keras")

    raw.to_csv(
        metric_dir / "raw_validation_predictions.csv",
        index_label="hour_utc",
    )
    bounded.to_csv(
        metric_dir / "validation_predictions.csv",
        index_label="hour_utc",
    )
    pd.concat(
        [
            raw_scores.assign(variant="raw"),
            bounded_scores.assign(variant="bounded"),
        ],
        ignore_index=True,
    ).to_csv(metric_dir / "raw_and_bounded_scores.csv", index=False)

    comparison.to_csv(
        metric_dir / "baseline_comparison.csv",
        index=False,
    )

    history_frame.index = np.arange(1, len(history_frame) + 1)
    history_frame.to_csv(
        metric_dir / "training_history.csv",
        index_label="epoch",
    )

    metadata = {
        **summary,
        "input_columns": [metric],
        "lookback_hours": LOOKBACK,
        "input_window": "t-71 through t inclusive",
        "target_columns": columns,
        "training_mean": mean,
        "training_scale": scale,
        "bounds": list(BOUNDS[metric]),
        "scaling_period_start": train.index.min().isoformat(),
        "scaling_period_end": train.index.max().isoformat(),
        "training_origin_start": train_y.index.min().isoformat(),
        "training_origin_end": train_y.index.max().isoformat(),
        "validation_origin_start": val_y.index.min().isoformat(),
        "validation_origin_end": val_y.index.max().isoformat(),
        "lstm_units": 32,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "seed": SEED,
        "tensorflow_version": tf.__version__,
        "epoch_selection": "Minimum validation scaled MSE before clipping",
        "reported_scores": "Physical units after applying fixed bounds",
        "inference_steps": [
            "Normalize the 72 input values using training_mean and training_scale",
            "Run the saved model",
            "Multiply outputs by training_scale and add training_mean",
            "Apply the saved bounds",
        ],
        "availability_assumption": (
            "Observation at time t is available at time t. "
            "Actual publication delays are not simulated."
        ),
        "test_data_used": False,
    }

    with (metric_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    print("\nBounded LSTM results at +24 hours:")
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
    print(f"LSTM:     {lstm_rmse:.4f}")
    print(f"Baseline: {baseline_rmse:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Predictions clipped: {changed_values}/{values.size}")
    print(f"Saved to: {metric_dir}", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train separate univariate LSTM forecasting models."
    )
    parser.add_argument(
        "--metric",
        choices=[*METRICS, "remaining", "all"],
        default="remaining",
        help="'remaining' trains all continuous metrics except temperature.",
    )
    args = parser.parse_args()

    if args.metric == "remaining":
        metrics = [
            metric for metric in METRICS
            if metric != "temperature_c"
        ]
    elif args.metric == "all":
        metrics = METRICS
    else:
        metrics = [args.metric]

    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        data = load_data()

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_dir = DATA_DIR / "models" / f"lstm_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)

        print(f"Metrics: {', '.join(metrics)}")
        print("One model will train at a time.")
        print(f"Run directory: {run_dir}")
        print("Existing model folders will be preserved.")

        summaries = []

        for metric in metrics:
            summary = train_metric(metric, data, run_dir)
            summaries.append(summary)

            # Preserve completed results even if a later metric fails.
            pd.DataFrame(summaries).to_csv(
                run_dir / "summary.csv",
                index=False,
            )

        print("\nCompleted LSTM comparisons:")
        print(
            pd.DataFrame(summaries).round(4).to_string(index=False)
        )
        print(f"\nSaved to: {run_dir}")
        print("Validation was used for early stopping.")
        print("These are provisional validation results.")
        print("No existing model replaced.")
        print("Test data not read. No Hopsworks upload.")
        return 0

    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        tf.errors.OpError,
    ) as error:
        print(f"\nLSTM training failed: {error}")
        print("Any previously completed metric results remain saved.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())