import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_statistical import load_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HORIZONS = range(1, 25)
RAIN_THRESHOLD_MM = 0.1
DECISION_THRESHOLD = 0.5

TARGET_COLUMNS = [
    f"rain_event_t_plus_{horizon}h"
    for horizon in HORIZONS
]

RF_PARAMETERS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": 2,
}


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
    )[TARGET_COLUMNS]

    if not x.index.equals(y.index):
        raise ValueError(f"Input and target timestamps differ: {name}.")

    if not x.columns.is_unique:
        raise ValueError(f"Duplicate input columns: {name}.")

    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid input features: {name}.")

    if not np.isin(y.to_numpy(), [0, 1]).all():
        raise ValueError(f"Rain targets must be binary: {name}.")

    return x, y.astype(int)


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

    rain = pd.concat(
        [train_raw["rain_mm"], val_raw["rain_mm"]]
    )

    expected = pd.date_range(
        "2025-11-01",
        "2026-10-01",
        inclusive="left",
        freq="h",
        tz="UTC",
        name="hour_utc",
    )
    if not rain.index.equals(expected):
        raise ValueError("Rain history is not continuous.")

    values = rain.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid historical rain amounts.")

    for targets in (y_train, y_val):
        for horizon, column in zip(
            HORIZONS, TARGET_COLUMNS, strict=True
        ):
            future_times = targets.index + pd.Timedelta(hours=horizon)
            expected_events = (
                rain.loc[future_times].to_numpy() >= RAIN_THRESHOLD_MM
            ).astype(int)

            if not np.array_equal(
                expected_events, targets[column].to_numpy()
            ):
                raise ValueError(f"Target alignment failed: {column}.")

    return x_train, y_train, x_val, y_val, rain


def probability_scores(actual, probabilities):
    actual = np.asarray(actual, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if (
        probabilities.shape != actual.shape
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid probabilities.")

    predicted = probabilities >= DECISION_THRESHOLD

    tp = int(((actual == 1) & predicted).sum())
    fp = int(((actual == 0) & predicted).sum())
    fn = int(((actual == 1) & ~predicted).sum())

    denominator = 2 * tp + fp + fn

    return {
        "actual_rain_hours": int(actual.sum()),
        "predicted_rain_hours": int(predicted.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_probability": float(probabilities.mean()),
        "brier_score": float(np.mean((probabilities - actual) ** 2)),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / denominator if denominator else np.nan,
    }


def reliability_rows(actual, probabilities, horizon, model_name):
    bins = np.minimum(
        (np.asarray(probabilities) * 10).astype(int), 9
    )
    rows = []

    for number in range(10):
        mask = bins == number
        count = int(mask.sum())

        rows.append(
            {
                "model": model_name,
                "horizon_hours": horizon,
                "bin_lower": number / 10,
                "bin_upper": (number + 1) / 10,
                "count": count,
                "mean_probability": (
                    float(probabilities[mask].mean()) if count else np.nan
                ),
                "observed_rain_fraction": (
                    float(actual[mask].mean()) if count else np.nan
                ),
            }
        )

    return rows


def load_previous_predictions(run_name, expected_index):
    path = (
        DATA_DIR / "models" / run_name / "validation_probabilities.csv"
    )

    predictions = pd.read_csv(path)
    predictions["hour_utc"] = pd.to_datetime(
        predictions["hour_utc"], utc=True, errors="raise"
    )
    predictions = predictions.set_index("hour_utc").sort_index()

    if not predictions.index.equals(expected_index):
        raise ValueError("Previous predictions use different timestamps.")

    if list(predictions.columns) != TARGET_COLUMNS:
        raise ValueError("Previous prediction columns do not match.")

    values = predictions.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Invalid saved logistic probabilities.")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate hourly rain probabilities."
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest"],
        default="logistic",
    )
    parser.add_argument(
        "--logistic-run",
        help="Existing logistic run folder to compare with Random Forest.",
    )
    args = parser.parse_args()

    if args.model == "random_forest" and not args.logistic_run:
        parser.error("--logistic-run is required for Random Forest.")

    try:
        x_train, y_train, x_val, y_val, rain = load_data()

        previous = None
        if args.model == "random_forest":
            previous = load_previous_predictions(
                args.logistic_run, x_val.index
            )

        scaler = None
        if args.model == "logistic":
            scaler = StandardScaler()
            train_inputs = scaler.fit_transform(x_train)
            val_inputs = scaler.transform(x_val)
        else:
            train_inputs = x_train
            val_inputs = x_val

        print(f"Model: {args.model}")
        print(f"Training examples: {len(x_train)}")
        print(f"Validation examples: {len(x_val)}")
        print(f"Input features: {x_train.shape[1]}")
        print("Training one classifier per forecast horizon.")

        models = {}
        frequencies = {}
        probability_columns = {}
        score_rows = []
        calibration_rows = []

        for horizon, column in zip(
            HORIZONS, TARGET_COLUMNS, strict=True
        ):
            labels = y_train[column].to_numpy()
            actual = y_val[column].to_numpy()
            frequency = float(labels.mean())
            frequencies[column] = frequency

            print(
                f"\nTraining +{horizon}h — "
                f"rain examples: {int(labels.sum())}",
                flush=True,
            )

            if np.unique(labels).size < 2:
                model = None
                probabilities = np.full(len(x_val), frequency)
            else:
                if args.model == "random_forest":
                    model = RandomForestClassifier(**RF_PARAMETERS)
                else:
                    model = LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight=None,
                    )

                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(train_inputs, labels)

                positive_index = int(
                    np.flatnonzero(model.classes_ == 1)[0]
                )
                probabilities = model.predict_proba(val_inputs)[
                    :, positive_index
                ]

            models[column] = model
            probability_columns[column] = probabilities

            previous_times = (
                x_val.index + pd.Timedelta(hours=horizon - 24)
            )
            previous_day = (
                rain.loc[previous_times].to_numpy() >= RAIN_THRESHOLD_MM
            ).astype(float)

            candidates = {
                args.model: probabilities,
                "always_dry": np.zeros(len(x_val)),
                "training_frequency": np.full(len(x_val), frequency),
                "previous_day": previous_day,
            }

            if previous is not None:
                candidates["logistic"] = previous[column].to_numpy()

            for name, candidate in candidates.items():
                score_rows.append(
                    {
                        "model": name,
                        "horizon_hours": horizon,
                        **probability_scores(actual, candidate),
                    }
                )
                calibration_rows.extend(
                    reliability_rows(actual, candidate, horizon, name)
                )

        scores = pd.DataFrame(score_rows)
        summary = (
            scores.groupby("model", as_index=False)["brier_score"]
            .mean()
            .rename(columns={"brier_score": "mean_brier_score"})
            .sort_values("mean_brier_score")
        )

        print("\nMean Brier score across 24 horizons — lower is better:")
        print(summary.round(6).to_string(index=False))

        print("\nRain-event evaluation at +24 hours:")
        print(
            scores.loc[
                scores["horizon_hours"].eq(24)
            ].round(4).to_string(index=False)
        )

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_dir = (
            DATA_DIR / "models" / f"rain_{args.model}_{run_id}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "model": args.model,
            "rain_event_definition": "rain_mm >= 0.1",
            "probability_meaning": "Rain during each individual target hour",
            "feature_columns": list(x_train.columns),
            "target_columns": TARGET_COLUMNS,
            "training_frequency_by_target": frequencies,
            "decision_threshold": DECISION_THRESHOLD,
            "parameters": (
                RF_PARAMETERS if args.model == "random_forest"
                else {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 2000,
                    "class_weight": None,
                }
            ),
            "training_origin_start": x_train.index.min().isoformat(),
            "training_origin_end": x_train.index.max().isoformat(),
            "validation_origin_start": x_val.index.min().isoformat(),
            "validation_origin_end": x_val.index.max().isoformat(),
            "logistic_comparison_run": args.logistic_run,
            "sklearn_version": sklearn.__version__,
            "calibration": "Not recalibrated",
            "test_data_used": False,
            "source_note": "Rain labels are ERA5 reanalysis estimates.",
            "availability_assumption": (
                "Features at time t are available at time t; "
                "publication delays are not simulated."
            ),
        }

        joblib.dump(
            {
                "scaler": scaler,
                "models_by_target": models,
                "metadata": metadata,
            },
            run_dir / "candidate.joblib",
            compress=3,
        )

        pd.DataFrame(
            probability_columns, index=x_val.index
        ).to_csv(
            run_dir / "validation_probabilities.csv",
            index_label="hour_utc",
        )
        scores.to_csv(run_dir / "scores_by_horizon.csv", index=False)
        summary.to_csv(run_dir / "summary.csv", index=False)
        pd.DataFrame(calibration_rows).to_csv(
            run_dir / "reliability_by_horizon.csv",
            index=False,
        )

        with (run_dir / "metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2, allow_nan=False)

        print(f"\nSaved to: {run_dir}")
        print("Threshold fixed at 0.5; not tuned.")
        print("Probability calibration is not yet established.")
        print("Existing models preserved.")
        print("Test data not read. No Hopsworks upload.")
        return 0

    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        ConvergenceWarning,
    ) as error:
        print(f"\nRain-probability training failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())