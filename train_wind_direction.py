import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_statistical import load_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ALPHA = 1000.0
VECTOR_EPSILON = 1e-8

# Match the project's current direction-validity definition.
CALM_THRESHOLD_MS = 0.0


def direction_columns(horizon):
    return [
        f"wind_direction_sin_t_plus_{horizon}h",
        f"wind_direction_cos_t_plus_{horizon}h",
    ]


def load_supervised(name, start, end):
    x = load_frame(
        DATA_DIR / "supervised" / f"{name}_X.csv",
        start,
        end,
    )
    y = load_frame(
        DATA_DIR / "supervised" / f"{name}_y.csv",
        start,
        end,
    )

    if not x.index.equals(y.index):
        raise ValueError(f"Input and target timestamps differ: {name}.")

    if not x.columns.is_unique:
        raise ValueError(f"Duplicate input columns: {name}.")

    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid input features: {name}.")

    return x, y


def load_data():
    x_train, y_train = load_supervised(
        "train", "2025-11-08", "2026-07-31"
    )
    x_val, y_val = load_supervised(
        "validation", "2026-08-01", "2026-09-30"
    )

    if list(x_train.columns) != list(x_val.columns):
        raise ValueError("Feature columns differ between splits.")

    if (
        x_train.index.max() + pd.Timedelta(hours=24)
        >= x_val.index.min()
    ):
        raise ValueError("Training targets overlap validation.")

    train_raw = load_frame(
        DATA_DIR / "splits" / "train.csv",
        "2025-11-01",
        "2026-08-01",
    )
    val_raw = load_frame(
        DATA_DIR / "splits" / "validation.csv",
        "2026-08-01",
        "2026-10-01",
    )

    columns = ["wind_direction_deg", "wind_speed_ms"]
    history = pd.concat([train_raw[columns], val_raw[columns]])

    expected = pd.date_range(
        "2025-11-01",
        "2026-10-01",
        inclusive="left",
        freq="h",
        tz="UTC",
        name="hour_utc",
    )

    if not history.index.equals(expected):
        raise ValueError("Wind history is not continuous.")

    if not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError("Invalid historical wind values.")

    if (history["wind_speed_ms"] < 0).any():
        raise ValueError("Negative historical wind speed.")

    if not history["wind_direction_deg"].between(0, 360).all():
        raise ValueError("Historical direction outside 0–360 degrees.")

    return x_train, y_train, x_val, y_val, history


def validate_targets(targets, history, horizon):
    columns = direction_columns(horizon)
    speed_column = f"wind_speed_ms_t_plus_{horizon}h"

    future_times = targets.index + pd.Timedelta(hours=horizon)
    future = history.loc[future_times]

    speeds = targets[speed_column].to_numpy(dtype=float)

    if not np.allclose(
        speeds,
        future["wind_speed_ms"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(f"Speed target alignment failed at +{horizon}h.")

    valid = speeds > CALM_THRESHOLD_MS
    components = targets[columns].to_numpy(dtype=float)

    if not np.isfinite(components[valid]).all():
        raise ValueError(f"Invalid direction targets at +{horizon}h.")

    radians = np.deg2rad(
        future["wind_direction_deg"].to_numpy(dtype=float)
    )
    expected_components = np.column_stack(
        [np.sin(radians), np.cos(radians)]
    )

    if not np.allclose(
        components[valid],
        expected_components[valid],
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Sine/cosine target alignment failed at +{horizon}h."
        )

    return valid, components, future["wind_direction_deg"].to_numpy()


def angular_scores(actual, predicted):
    if len(actual) == 0:
        return {
            "angular_mae_deg": np.nan,
            "angular_rmse_deg": np.nan,
        }

    # Shortest signed difference, in [-180, 180).
    errors = (predicted - actual + 180.0) % 360.0 - 180.0

    return {
        "angular_mae_deg": float(np.mean(np.abs(errors))),
        "angular_rmse_deg": float(np.sqrt(np.mean(errors ** 2))),
    }


def main():
    try:
        x_train, y_train, x_val, y_val, history = load_data()

        print(f"Training examples: {len(x_train)}")
        print(f"Validation examples: {len(x_val)}")
        print(f"Input features: {x_train.shape[1]}")
        print(f"Ridge alpha: {ALPHA}")
        print("Predicting sine and cosine for each forecast horizon.")

        models = {}
        prediction_columns = {}
        diagnostic_columns = {}
        score_rows = []
        training_counts = {}

        for horizon in range(1, 25):
            train_valid, train_components, _ = validate_targets(
                y_train, history, horizon
            )
            val_valid, _, actual_direction = validate_targets(
                y_val, history, horizon
            )

            count = int(train_valid.sum())
            if count < 2:
                raise ValueError(
                    f"Insufficient non-calm training rows at +{horizon}h."
                )

            training_counts[str(horizon)] = count

            print(
                f"\nTraining +{horizon}h: {count} non-calm examples",
                flush=True,
            )

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=ALPHA, solver="svd")),
                ]
            )

            # Future speed is used only to identify valid training labels.
            # It is never added to the input features.
            model.fit(
                x_train.loc[train_valid],
                train_components[train_valid],
            )

            predicted_components = model.predict(x_val)

            if not np.isfinite(predicted_components).all():
                raise ValueError("Non-finite predicted components.")

            sin_prediction = predicted_components[:, 0]
            cos_prediction = predicted_components[:, 1]
            magnitude = np.hypot(sin_prediction, cos_prediction)

            prediction_valid = magnitude > VECTOR_EPSILON

            predicted_direction = (
                np.rad2deg(
                    np.arctan2(sin_prediction, cos_prediction)
                ) % 360.0
            )
            predicted_direction[~prediction_valid] = np.nan

            # Previous-day direction at the matching target hour.
            previous_times = (
                x_val.index + pd.Timedelta(hours=horizon - 24)
            )
            previous = history.loc[previous_times]

            baseline_direction = previous[
                "wind_direction_deg"
            ].to_numpy(dtype=float) % 360.0

            baseline_valid = (
                previous["wind_speed_ms"].to_numpy()
                > CALM_THRESHOLD_MS
            )

            # Both methods receive exactly the same evaluation rows.
            common = val_valid & baseline_valid & prediction_valid

            shared_counts = {
                "horizon_hours": horizon,
                "evaluated_rows": int(common.sum()),
                "excluded_target_calm": int((~val_valid).sum()),
                "excluded_baseline_calm_after_target_filter": int(
                    (val_valid & ~baseline_valid).sum()
                ),
                "excluded_zero_vector_after_calm_filters": int(
                    (
                        val_valid
                        & baseline_valid
                        & ~prediction_valid
                    ).sum()
                ),
            }

            for name, predictions in (
                ("ridge", predicted_direction),
                ("previous_day", baseline_direction),
            ):
                score_rows.append(
                    {
                        "model": name,
                        **shared_counts,
                        **angular_scores(
                            actual_direction[common],
                            predictions[common],
                        ),
                    }
                )

            models[str(horizon)] = model
            prediction_columns[
                f"wind_direction_deg_t_plus_{horizon}h"
            ] = predicted_direction

            diagnostic_columns[
                f"predicted_sin_t_plus_{horizon}h"
            ] = sin_prediction
            diagnostic_columns[
                f"predicted_cos_t_plus_{horizon}h"
            ] = cos_prediction
            diagnostic_columns[
                f"vector_magnitude_t_plus_{horizon}h"
            ] = magnitude
            diagnostic_columns[
                f"evaluation_valid_t_plus_{horizon}h"
            ] = common

        scores = pd.DataFrame(score_rows)

        summary = (
            scores.groupby("model", as_index=False)
            .agg(
                mean_angular_mae_deg=("angular_mae_deg", "mean"),
                mean_angular_rmse_deg=("angular_rmse_deg", "mean"),
                horizons_evaluated=("angular_mae_deg", "count"),
            )
            .sort_values("mean_angular_mae_deg")
        )

        print("\nMean angular errors across 24 horizons:")
        print(summary.round(4).to_string(index=False))

        print("\nWind-direction evaluation at +24 hours:")
        print(
            scores.loc[
                scores["horizon_hours"].eq(24)
            ].round(4).to_string(index=False)
        )

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_dir = DATA_DIR / "models" / f"wind_direction_ridge_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "model": "Ridge on sine and cosine",
            "alpha": ALPHA,
            "alpha_tuned": False,
            "feature_columns": list(x_train.columns),
            "component_order": ["sin", "cos"],
            "direction_conversion": (
                "degrees(arctan2(predicted_sin, predicted_cos)) modulo 360"
            ),
            "calm_threshold_ms": CALM_THRESHOLD_MS,
            "vector_epsilon": VECTOR_EPSILON,
            "training_rows_by_horizon": training_counts,
            "training_origin_start": x_train.index.min().isoformat(),
            "training_origin_end": x_train.index.max().isoformat(),
            "validation_origin_start": x_val.index.min().isoformat(),
            "validation_origin_end": x_val.index.max().isoformat(),
            "sklearn_version": sklearn.__version__,
            "evaluation": (
                "Common rows with non-calm target and baseline wind "
                "and nonzero predicted direction vector."
            ),
            "vector_magnitude_note": (
                "Diagnostic only; not a calibrated confidence score."
            ),
            "operational_note": (
                "Actual future speed is used only for label validity "
                "and evaluation. Live direction display must use a "
                "separate predicted-speed or calm policy."
            ),
            "availability_assumption": (
                "Inputs at time t are available at time t; "
                "publication delays are not simulated."
            ),
            "test_data_used": False,
        }

        joblib.dump(
            {
                "pipelines_by_horizon": models,
                "metadata": metadata,
            },
            run_dir / "candidate.joblib",
            compress=3,
        )

        pd.DataFrame(
            prediction_columns, index=x_val.index
        ).to_csv(
            run_dir / "validation_predictions.csv",
            index_label="hour_utc",
        )

        pd.DataFrame(
            diagnostic_columns, index=x_val.index
        ).to_csv(
            run_dir / "validation_diagnostics.csv",
            index_label="hour_utc",
        )

        scores.to_csv(run_dir / "scores_by_horizon.csv", index=False)
        summary.to_csv(run_dir / "summary.csv", index=False)

        with (run_dir / "metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2, allow_nan=False)

        print(f"\nSaved to: {run_dir}")
        print("Lower angular error is better.")
        print("Calm-label masks were not included as input features.")
        print("Undefined predictions or scores are saved as blank.")
        print("Existing models preserved.")
        print("Test data not read. No Hopsworks upload.")
        return 0

    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"\nWind-direction training failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())