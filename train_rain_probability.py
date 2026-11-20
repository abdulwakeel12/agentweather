import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
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

    # Check that labels represent rain at exactly t + horizon.
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
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Probabilities must be between zero and one.")

    predicted = probabilities >= DECISION_THRESHOLD

    tp = int(((actual == 1) & predicted).sum())
    fp = int(((actual == 0) & predicted).sum())
    fn = int(((actual == 1) & ~predicted).sum())

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else np.nan

    return {
        "actual_rain_hours": int(actual.sum()),
        "predicted_rain_hours": int(predicted.sum()),
        "mean_probability": float(probabilities.mean()),
        "brier_score": float(np.mean((probabilities - actual) ** 2)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def reliability_rows(actual, probabilities, horizon):
    # Fixed bins; these summarize probabilities, not recalibrate them.
    bins = np.minimum(
        (np.asarray(probabilities) * 10).astype(int), 9
    )
    rows = []

    for number in range(10):
        mask = bins == number
        count = int(mask.sum())

        rows.append(
            {
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


def main():
    try:
        x_train, y_train, x_val, y_val, rain = load_data()

        print(f"Training examples: {len(x_train)}")
        print(f"Validation examples: {len(x_val)}")
        print(f"Input features: {x_train.shape[1]}")
        print(f"Rain event: rain_mm >= {RAIN_THRESHOLD_MM}")
        print("Training one classifier per forecast horizon.")

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(x_train)
        val_scaled = scaler.transform(x_val)

        models = {}
        training_frequencies = {}
        probability_columns = {}
        baseline_columns = {}
        score_rows = []
        calibration_rows = []

        for horizon, column in zip(
            HORIZONS, TARGET_COLUMNS, strict=True
        ):
            labels = y_train[column].to_numpy()
            actual = y_val[column].to_numpy()
            frequency = float(labels.mean())

            training_frequencies[column] = frequency

            print(
                f"\nHorizon +{horizon}h — "
                f"training rain examples: {int(labels.sum())}",
                flush=True,
            )

            if np.unique(labels).size < 2:
                # A classifier cannot learn two classes from one class.
                model = None
                probabilities = np.full(len(x_val), frequency)
                print("Single training class; using constant probability.")
            else:
                model = LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,
                )

                # Fail explicitly rather than accept an unconverged fit.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(train_scaled, labels)

                positive_index = int(
                    np.flatnonzero(model.classes_ == 1)[0]
                )
                probabilities = model.predict_proba(val_scaled)[
                    :, positive_index
                ]

            models[column] = model
            probability_columns[column] = probabilities

            # Previous-day value for target t+h is at t+h-24.
            # This is never later than the prediction time t.
            previous_times = (
                x_val.index + pd.Timedelta(hours=horizon - 24)
            )
            previous_day = (
                rain.loc[previous_times].to_numpy() >= RAIN_THRESHOLD_MM
            ).astype(float)

            candidates = {
                "logistic": probabilities,
                "always_dry": np.zeros(len(x_val)),
                "training_frequency": np.full(len(x_val), frequency),
                "previous_day": previous_day,
            }

            for name, candidate in candidates.items():
                score_rows.append(
                    {
                        "model": name,
                        "horizon_hours": horizon,
                        **probability_scores(actual, candidate),
                    }
                )
                if name != "logistic":
                    baseline_columns[f"{name}_{column}"] = candidate

            calibration_rows.extend(
                reliability_rows(actual, probabilities, horizon)
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
        run_dir = DATA_DIR / "models" / f"rain_logistic_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "rain_event_definition": "rain_mm >= 0.1",
            "probability_meaning": (
                "Rain event during the individual target hour; "
                "not any rain across the entire next 24 hours."
            ),
            "feature_columns": list(x_train.columns),
            "target_columns": TARGET_COLUMNS,
            "training_frequency_by_target": training_frequencies,
            "decision_threshold": DECISION_THRESHOLD,
            "C": 1.0,
            "class_weight": None,
            "training_origin_start": x_train.index.min().isoformat(),
            "training_origin_end": x_train.index.max().isoformat(),
            "validation_origin_start": x_val.index.min().isoformat(),
            "validation_origin_end": x_val.index.max().isoformat(),
            "sklearn_version": sklearn.__version__,
            "calibration": "Not recalibrated; reliability tables saved",
            "test_data_used": False,
            "source_note": (
                "Rain labels come from historical ERA5 reanalysis. "
                "Operational source compatibility still requires validation."
            ),
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
        )

        pd.DataFrame(
            probability_columns, index=x_val.index
        ).to_csv(
            run_dir / "validation_probabilities.csv",
            index_label="hour_utc",
        )

        pd.DataFrame(
            baseline_columns, index=x_val.index
        ).to_csv(
            run_dir / "baseline_probabilities.csv",
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
        print("Classification threshold fixed at 0.5; not tuned.")
        print("Undefined precision or recall is saved as blank.")
        print("Probability calibration is not yet established.")
        print("No existing model replaced.")
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