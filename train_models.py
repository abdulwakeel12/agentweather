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

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]

TARGET_COLUMNS = [
    f"{metric}_t_plus_{horizon}h"
    for horizon in range(1, 25)
    for metric in METRICS
]

# Fixed physical constraints, not learned from validation data.
BOUNDS = {
    "us_aqi": (0, None),
    "temperature_c": (None, None),
    "precipitation_mm": (0, None),
    "humidity_pct": (0, 100),
    "uv_index": (0, None),
    "wind_speed_ms": (0, None),
}


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

        frames.append(df)

    x, y = frames
    y = y[TARGET_COLUMNS]

    for frame in (x, y):
        if not frame.columns.is_unique:
            raise ValueError(f"Duplicate columns in {name}.")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or invalid values in {name}.")

    return x, y


def apply_bounds(predictions):
    bounded = predictions.copy()

    for metric in METRICS:
        columns = [
            f"{metric}_t_plus_{horizon}h"
            for horizon in range(1, 25)
        ]
        lower, upper = BOUNDS[metric]
        if lower is not None or upper is not None:
            bounded[columns] = bounded[columns].clip(
                lower=lower, upper=upper
            )

    return bounded


def main():
    try:
        x_train, y_train = load_split(
            "train", "2025-11-08", "2026-07-31"
        )
        x_val, y_val = load_split(
            "validation", "2026-08-01", "2026-09-30"
        )

        if list(x_train.columns) != list(x_val.columns):
            raise ValueError("Feature columns differ between splits.")

        if x_train.index.max() + pd.Timedelta(hours=24) >= x_val.index.min():
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

        if not np.isfinite(baseline["rmse"].to_numpy(dtype=float)).all():
            raise ValueError("Invalid baseline RMSE values.")

        models = {}
        predictions_by_alpha = {}
        scores = []

        print(f"Training rows: {len(x_train)}")
        print(f"Validation rows: {len(x_val)}")

        for alpha in ALPHAS:
            print(f"\nTraining Ridge: alpha={alpha}", flush=True)

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha, solver="svd")),
                ]
            )
            model.fit(x_train, y_train)

            raw = pd.DataFrame(
                model.predict(x_val),
                index=x_val.index,
                columns=TARGET_COLUMNS,
            )

            if not np.isfinite(raw.to_numpy()).all():
                raise ValueError("Non-finite model predictions.")

            bounded = apply_bounds(raw)
            models[alpha] = model
            predictions_by_alpha[alpha] = bounded

            for variant, predictions in (
                ("raw", raw),
                ("bounded", bounded),
            ):
                for metric in METRICS:
                    for horizon in range(1, 25):
                        column = f"{metric}_t_plus_{horizon}h"
                        scores.append(
                            {
                                "alpha": alpha,
                                "variant": variant,
                                "metric": metric,
                                "horizon_hours": horizon,
                                **regression_scores(
                                    y_val[column].to_numpy(),
                                    predictions[column].to_numpy(),
                                ),
                            }
                        )

        scores = pd.DataFrame(scores)

        # Each metric is selected separately: never average different units.
        averages = (
            scores.loc[scores["variant"].eq("bounded")]
            .groupby(["metric", "alpha"], as_index=False)["rmse"]
            .mean()
            .rename(columns={"rmse": "mean_horizon_rmse"})
        )

        best = (
            averages.sort_values(
                ["metric", "mean_horizon_rmse", "alpha"]
            )
            .drop_duplicates("metric")
            .set_index("metric")
        )

        baseline_means = baseline.groupby("metric")["rmse"].mean()

        summary_rows = []
        selected_scores = []
        prediction_parts = []
        selected_alphas = {}

        for metric in METRICS:
            alpha = float(best.loc[metric, "alpha"])
            ridge_rmse = float(best.loc[metric, "mean_horizon_rmse"])
            baseline_rmse = float(baseline_means.loc[metric])

            selected_alphas[metric] = alpha

            columns = [
                f"{metric}_t_plus_{horizon}h"
                for horizon in range(1, 25)
            ]
            prediction_parts.append(
                predictions_by_alpha[alpha][columns]
            )

            selected_scores.append(
                scores.loc[
                    scores["variant"].eq("bounded")
                    & scores["metric"].eq(metric)
                    & scores["alpha"].eq(alpha)
                ]
            )

            summary_rows.append(
                {
                    "metric": metric,
                    "selected_alpha": alpha,
                    "ridge_mean_rmse": ridge_rmse,
                    "baseline_mean_rmse": baseline_rmse,
                    "provisional_choice": (
                        "ridge" if ridge_rmse < baseline_rmse else "baseline"
                    ),
                }
            )

        selected_predictions = pd.concat(
            prediction_parts, axis=1
                ).loc[:, TARGET_COLUMNS]

        summary = pd.DataFrame(summary_rows)
        selected_scores = pd.concat(selected_scores, ignore_index=True)

        comparison = selected_scores.merge(
            baseline[["metric", "horizon_hours", "mae", "rmse", "r2"]],
            on=["metric", "horizon_hours"],
            validate="one_to_one",
            suffixes=("_ridge", "_baseline"),
        )

        print("\nSelection using mean RMSE across 24 horizons:")
        print(summary.round(4).to_string(index=False))

        print("\nSelected bounded Ridge results at +24 hours:")
        print(
            comparison.loc[
                comparison["horizon_hours"].eq(24),
                [
                    "metric",
                    "alpha",
                    "mae_ridge",
                    "rmse_ridge",
                    "rmse_baseline",
                    "r2_ridge",
                ],
            ].round(4).to_string(index=False)
        )

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_dir = DATA_DIR / "models" / f"ridge_tuned_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)

        # Keep the selected pipelines, plus the metadata needed for inference.
        selected_models = {
            alpha: models[alpha]
            for alpha in sorted(set(selected_alphas.values()))
        }

        joblib.dump(
            {
                "pipelines_by_alpha": selected_models,
                "selected_alpha_by_metric": selected_alphas,
                "feature_columns": list(x_train.columns),
                "target_columns": TARGET_COLUMNS,
                "bounds": BOUNDS,
                "selection_rule": "mean validation RMSE across 24 horizons",
                "sklearn_version": sklearn.__version__,
                "provisional_choices": summary.to_dict("records"),
            },
            run_dir / "candidate.joblib",
        )

        scores.to_csv(run_dir / "all_raw_and_bounded_scores.csv", index=False)
        summary.to_csv(run_dir / "selection_summary.csv", index=False)
        comparison.to_csv(run_dir / "selected_comparison.csv", index=False)
        selected_predictions.to_csv(
            run_dir / "selected_ridge_validation_predictions.csv",
            index_label="hour_utc",
        )

        print(f"\nSaved to: {run_dir}")
        print("Saved predictions are bounded Ridge candidates only.")
        print("No production model replaced. No test data read.")
        print("No Hopsworks upload.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"Training failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and close open output files.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()