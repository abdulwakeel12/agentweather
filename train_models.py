import argparse
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
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

BOUNDS = {
    "us_aqi": (0, None),
    "temperature_c": (None, None),
    "precipitation_mm": (0, None),
    "humidity_pct": (0, 100),
    "uv_index": (0, None),
    "wind_speed_ms": (0, None),
}

RF_PARAMETERS = {
    "n_estimators": 150,
    "max_depth": 14,
    "min_samples_leaf": 4,
    "max_features": 0.5,
    "random_state": 42,
    "n_jobs": 2,
}


def metric_columns(metric):
    return [
        f"{metric}_t_plus_{horizon}h"
        for horizon in range(1, 25)
    ]


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


def load_training_data():
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

    print(f"Training rows: {len(x_train)}")
    print(f"Validation rows: {len(x_val)}")

    return x_train, y_train, x_val, y_val


def validate_score_table(table, rmse_column):
    expected_keys = {
        (metric, horizon)
        for metric in METRICS
        for horizon in range(1, 25)
    }
    actual_keys = set(
        zip(table["metric"], table["horizon_hours"], strict=True)
    )

    if len(table) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("Comparison scores are incomplete or duplicated.")

    values = table[rmse_column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid values in {rmse_column}.")


def load_baseline():
    baseline = pd.read_csv(
        DATA_DIR / "evaluation" / "baseline_regression_by_horizon.csv"
    )
    validate_score_table(baseline, "rmse")
    return baseline


def apply_bounds(predictions):
    parts = []

    for metric in METRICS:
        values = predictions[metric_columns(metric)]
        lower, upper = BOUNDS[metric]

        if lower is not None or upper is not None:
            values = values.clip(lower=lower, upper=upper)

        parts.append(values)

    return pd.concat(parts, axis=1).loc[:, TARGET_COLUMNS]


def evaluate_predictions(actual, predictions):
    if not actual.index.equals(predictions.index):
        raise ValueError("Prediction and target timestamps differ.")

    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite model predictions.")

    rows = []

    for metric in METRICS:
        for horizon in range(1, 25):
            column = f"{metric}_t_plus_{horizon}h"
            rows.append(
                {
                    "metric": metric,
                    "horizon_hours": horizon,
                    **regression_scores(
                        actual[column].to_numpy(),
                        predictions[column].to_numpy(),
                    ),
                }
            )

    return pd.DataFrame(rows)


def create_run_directory(prefix):
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    run_dir = DATA_DIR / "models" / f"{prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def model_metadata(x_train, x_val):
    return {
        "feature_columns": list(x_train.columns),
        "target_columns": TARGET_COLUMNS,
        "bounds": BOUNDS,
        "sklearn_version": sklearn.__version__,
        "training_start": x_train.index.min().isoformat(),
        "training_end": x_train.index.max().isoformat(),
        "validation_start": x_val.index.min().isoformat(),
        "validation_end": x_val.index.max().isoformat(),
    }


def train_ridge():
    x_train, y_train, x_val, y_val = load_training_data()
    baseline = load_baseline()

    models = {}
    predictions_by_alpha = {}
    score_parts = []

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
            scores = evaluate_predictions(y_val, predictions)
            scores["alpha"] = alpha
            scores["variant"] = variant
            score_parts.append(scores)

    scores = pd.concat(score_parts, ignore_index=True)

    averages = (
        scores.loc[scores["variant"].eq("bounded")]
        .groupby(["metric", "alpha"], as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "mean_horizon_rmse"})
    )

    best = (
        averages.sort_values(["metric", "mean_horizon_rmse", "alpha"])
        .drop_duplicates("metric")
        .set_index("metric")
    )

    baseline_means = baseline.groupby("metric")["rmse"].mean()

    summary_rows = []
    selected_score_parts = []
    prediction_parts = []
    selected_alphas = {}

    for metric in METRICS:
        alpha = float(best.loc[metric, "alpha"])
        ridge_rmse = float(best.loc[metric, "mean_horizon_rmse"])
        baseline_rmse = float(baseline_means.loc[metric])

        selected_alphas[metric] = alpha
        prediction_parts.append(
            predictions_by_alpha[alpha][metric_columns(metric)]
        )

        selected_score_parts.append(
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
    selected_scores = pd.concat(selected_score_parts, ignore_index=True)

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

    run_dir = create_run_directory("ridge_tuned")

    selected_models = {
        alpha: models[alpha]
        for alpha in sorted(set(selected_alphas.values()))
    }

    joblib.dump(
        {
            **model_metadata(x_train, x_val),
            "pipelines_by_alpha": selected_models,
            "selected_alpha_by_metric": selected_alphas,
            "selection_rule": "mean validation RMSE across 24 horizons",
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


def train_random_forest(ridge_run):
    x_train, y_train, x_val, y_val = load_training_data()
    baseline = load_baseline()

    ridge_dir = DATA_DIR / "models" / ridge_run
    ridge_predictions = pd.read_csv(
        ridge_dir / "selected_ridge_validation_predictions.csv"
    )
    ridge_predictions["hour_utc"] = pd.to_datetime(
        ridge_predictions["hour_utc"], utc=True, errors="raise"
    )
    ridge_predictions = ridge_predictions.set_index("hour_utc").sort_index()

    if not ridge_predictions.index.equals(x_val.index):
        raise ValueError("Ridge validation timestamps differ.")

    ridge_predictions = ridge_predictions[TARGET_COLUMNS]

    # Recalculate Ridge scores against the current validation targets.
    ridge_scores = evaluate_predictions(y_val, ridge_predictions).rename(
        columns={"rmse": "rmse_ridge"}
    )

    models = {}
    prediction_parts = []

    print("Training one forest per metric, with 24 outputs each.")

    for metric in METRICS:
        print(f"\nTraining Random Forest: {metric}", flush=True)

        columns = metric_columns(metric)
        model = RandomForestRegressor(**RF_PARAMETERS)
        model.fit(x_train, y_train[columns])

        predictions = pd.DataFrame(
            model.predict(x_val),
            index=x_val.index,
            columns=columns,
        )

        if not np.isfinite(predictions.to_numpy()).all():
            raise ValueError(f"Invalid predictions for {metric}.")

        models[metric] = model
        prediction_parts.append(predictions)

    raw_predictions = pd.concat(
        prediction_parts, axis=1
    ).loc[:, TARGET_COLUMNS]
    predictions = apply_bounds(raw_predictions)

    scores = evaluate_predictions(y_val, predictions)
    raw_scores = evaluate_predictions(y_val, raw_predictions)

    comparison = (
        scores.rename(
            columns={
                "mae": "mae_rf",
                "rmse": "rmse_rf",
                "r2": "r2_rf",
            }
        )
        .merge(
            ridge_scores[["metric", "horizon_hours", "rmse_ridge"]],
            on=["metric", "horizon_hours"],
            validate="one_to_one",
        )
        .merge(
            baseline[["metric", "horizon_hours", "rmse"]].rename(
                columns={"rmse": "rmse_baseline"}
            ),
            on=["metric", "horizon_hours"],
            validate="one_to_one",
        )
    )

    # Baseline first so exact ties prefer the simpler model.
    score_columns = ["rmse_baseline", "rmse_ridge", "rmse_rf"]

    if not np.isfinite(
        comparison[score_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid comparison RMSE values.")

    summary = (
        comparison.groupby("metric")[score_columns]
        .mean()
        .reindex(METRICS)
    )
    summary["provisional_choice"] = summary[score_columns].idxmin(axis=1).map(
        {
            "rmse_rf": "random_forest",
            "rmse_ridge": "ridge",
            "rmse_baseline": "baseline",
        }
    )

    print("\nMean validation RMSE across all 24 horizons:")
    print(summary.round(4).to_string())

    print("\nRandom Forest comparison at +24 hours:")
    print(
        comparison.loc[
            comparison["horizon_hours"].eq(24),
            [
                "metric",
                "mae_rf",
                "rmse_rf",
                "rmse_ridge",
                "rmse_baseline",
                "r2_rf",
            ],
        ].round(4).to_string(index=False)
    )

    run_dir = create_run_directory("random_forest")

    joblib.dump(
        {
            **model_metadata(x_train, x_val),
            "models_by_metric": models,
            "target_columns_by_metric": {
                metric: metric_columns(metric)
                for metric in METRICS
            },
            "parameters": RF_PARAMETERS,
            "ridge_comparison_run": ridge_run,
            "provisional_choices": summary.reset_index().to_dict("records"),
        },
        run_dir / "candidate.joblib",
        compress=3,
    )

    predictions.to_csv(
        run_dir / "validation_predictions.csv",
        index_label="hour_utc",
    )
    raw_scores.to_csv(run_dir / "raw_validation_scores.csv", index=False)
    comparison.to_csv(run_dir / "validation_comparison.csv", index=False)
    summary.to_csv(run_dir / "selection_summary.csv")

    print(f"\nCandidate and results saved to: {run_dir}")
    print("Saved predictions are Random Forest candidates only.")
    print("Existing Ridge models preserved.")


def main():
    parser = argparse.ArgumentParser(
        description="Train AgentWeather classical forecasting candidates."
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "random_forest"],
        default="ridge",
    )
    parser.add_argument(
        "--ridge-run",
        help="Existing Ridge folder name inside data/models.",
    )
    args = parser.parse_args()

    if args.model == "random_forest" and not args.ridge_run:
        parser.error("--ridge-run is required for Random Forest.")

    try:
        if args.model == "random_forest":
            train_random_forest(args.ridge_run)
        else:
            train_ridge()

        print("No production model replaced. No test data read.")
        print("No Hopsworks upload.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"Training failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print(
            "File access failed. Check input paths, the Ridge run folder, "
            "and whether output files are open."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()