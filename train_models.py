from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_baseline import regression_scores

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SUPERVISED_DIR = DATA_DIR / "supervised"

METRICS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
]

TARGET_COLUMNS = [
    f"{metric}_t_plus_{horizon}h"
    for horizon in range(1, 25)
    for metric in METRICS
]

# Fixed starting value; no hyperparameter search yet.
ALPHA = 1.0


def load_split(name, start, end):
    frames = []

    for suffix in ("X", "y"):
        path = SUPERVISED_DIR / f"{name}_{suffix}.csv"
        df = pd.read_csv(path)
        df["hour_utc"] = pd.to_datetime(
            df["hour_utc"], utc=True, errors="raise"
        )
        df = df.set_index("hour_utc").sort_index()

        expected = pd.date_range(
            start, end, inclusive="left", freq="h", tz="UTC"
        )
        if not df.index.equals(expected):
            raise ValueError(f"Unexpected timestamps in {path.name}.")

        if not df.columns.is_unique:
            raise ValueError(f"Duplicate columns in {path.name}.")

        frames.append(df)

    x, y = frames
    y = y[TARGET_COLUMNS]

    for frame in (x, y):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or invalid values in {name}.")

    return x, y


def main():
    try:
        x_train, y_train = load_split(
            "train", "2025-11-08", "2026-07-31"
        )
        x_validation, y_validation = load_split(
            "validation", "2026-08-01", "2026-09-30"
        )

        if list(x_train.columns) != list(x_validation.columns):
            raise ValueError("Training and validation features differ.")

        if (
            x_train.index.max() + pd.Timedelta(hours=24)
            >= x_validation.index.min()
        ):
            raise ValueError("Training targets overlap validation.")

        baseline = pd.read_csv(
            DATA_DIR / "evaluation" / "baseline_regression_by_horizon.csv"
        )

        expected_keys = {
            (metric, horizon)
            for metric in METRICS
            for horizon in range(1, 25)
        }
        actual_keys = set(
            zip(baseline["metric"], baseline["horizon_hours"], strict=True)
        )

        if len(baseline) != len(expected_keys) or actual_keys != expected_keys:
            raise ValueError("Baseline results are incomplete or duplicated.")

        print(f"Training examples: {len(x_train)}")
        print(f"Validation examples: {len(x_validation)}")
        print(f"Input features: {len(x_train.columns)}")
        print(f"Continuous targets: {len(TARGET_COLUMNS)}")
        print(f"\nTraining Ridge with alpha={ALPHA}...", flush=True)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ALPHA, solver="svd")),
            ]
        )

        model.fit(x_train, y_train)

        predictions = pd.DataFrame(
            model.predict(x_validation),
            index=x_validation.index,
            columns=TARGET_COLUMNS,
        )

        if not np.isfinite(predictions.to_numpy()).all():
            raise ValueError("Model generated non-finite predictions.")

        # Evaluate raw predictions first, without hiding invalid ranges.
        score_rows = []

        for horizon in range(1, 25):
            for metric in METRICS:
                column = f"{metric}_t_plus_{horizon}h"
                score_rows.append(
                    {
                        "metric": metric,
                        "horizon_hours": horizon,
                        **regression_scores(
                            y_validation[column].to_numpy(),
                            predictions[column].to_numpy(),
                        ),
                    }
                )

        scores = pd.DataFrame(score_rows)

        comparison = scores.merge(
            baseline[["metric", "horizon_hours", "mae", "rmse", "r2"]],
            on=["metric", "horizon_hours"],
            validate="one_to_one",
            suffixes=("_ridge", "_baseline"),
        )

        comparison["ridge_better_rmse"] = (
            comparison["rmse_ridge"] < comparison["rmse_baseline"]
        )

        print("\nValidation comparison at +24 hours:")
        print(
            comparison.loc[
                comparison["horizon_hours"].eq(24),
                [
                    "metric",
                    "mae_ridge",
                    "rmse_ridge",
                    "rmse_baseline",
                    "r2_ridge",
                    "ridge_better_rmse",
                ],
            ].round(4).to_string(index=False)
        )

        print("\nHorizons with lower RMSE than the baseline:")
        wins = comparison.groupby("metric")["ridge_better_rmse"].sum()
        for metric in METRICS:
            print(f"{metric}: {int(wins[metric])}/24")

        print("\nRaw predictions outside physical bounds:")
        invalid_rows = []

        for metric in METRICS:
            columns = [
                f"{metric}_t_plus_{horizon}h"
                for horizon in range(1, 25)
            ]
            values = predictions[columns].to_numpy()

            if metric == "temperature_c":
                continue

            invalid = values < 0
            if metric == "humidity_pct":
                invalid |= values > 100

            count = int(invalid.sum())
            print(f"{metric}: {count}/{values.size}")

            invalid_rows.append(
                {
                    "metric": metric,
                    "invalid_predictions": count,
                    "total_predictions": values.size,
                }
            )

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_dir = DATA_DIR / "models" / f"ridge_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)

        joblib.dump(
            {
                "pipeline": model,
                "feature_columns": list(x_train.columns),
                "target_columns": TARGET_COLUMNS,
                "sklearn_version": sklearn.__version__,
                "alpha": ALPHA,
                "prediction_postprocessing": "none",
                "training_start": x_train.index.min().isoformat(),
                "training_end": x_train.index.max().isoformat(),
                "validation_start": x_validation.index.min().isoformat(),
                "validation_end": x_validation.index.max().isoformat(),
            },
            run_dir / "candidate.joblib",
        )

        predictions.to_csv(
            run_dir / "validation_predictions.csv",
            index_label="hour_utc",
        )
        comparison.to_csv(run_dir / "validation_comparison.csv", index=False)
        pd.DataFrame(invalid_rows).to_csv(
            run_dir / "prediction_bounds.csv", index=False
        )

        print(f"\nCandidate model and results saved to: {run_dir}")
        print("No existing model replaced.")
        print("Test data not read. No Hopsworks upload.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"Training failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and close open output files.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()