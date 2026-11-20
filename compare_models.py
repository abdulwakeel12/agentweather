from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

METRICS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
]

RIDGE_RUN = "ridge_tuned_20261120T132247043738"
RF_RUN = "random_forest_20261120T135049251958"
LSTM_RUN = "lstm_20261120T175340139816"
LSTM_TEMPERATURE_RUN = "lstm_temperature_20261120T174746793623"
RAIN_RUN = "rain_random_forest_20261120T190019747084"
WIND_RUN = "wind_direction_ridge_20261120T191558239804"

SARIMAX_RUNS = {
    "temperature_c": "sarimax_20261120T141127482438",
    "us_aqi": "sarimax_20261120T141346296418",
    "precipitation_mm": "sarimax_20261120T141400830267",
    "humidity_pct": "sarimax_20261120T141425792172",
    "uv_index": "sarimax_20261120T141454205086",
    "wind_speed_ms": "sarimax_20261120T141517735339",
}


def validate_horizons(frame, group_columns):
    """Require exactly one row for every horizon in each group."""
    if frame.duplicated(group_columns + ["horizon_hours"]).any():
        raise ValueError("Duplicate model/metric/horizon results.")

    for key, group in frame.groupby(group_columns):
        if set(group["horizon_hours"]) != set(range(1, 25)):
            raise ValueError(f"Incomplete forecast horizons: {key}")


def read_scores(path, model, mae_column, rmse_column, r2_column):
    frame = pd.read_csv(path)

    required = [
        "metric",
        "horizon_hours",
        mae_column,
        rmse_column,
        r2_column,
    ]
    result = frame.loc[:, required].copy()
    result.columns = ["metric", "horizon_hours", "mae", "rmse", "r2"]
    result["model"] = model
    result["source_file"] = str(path.relative_to(BASE_DIR))

    if not np.isfinite(result[["mae", "rmse"]].to_numpy(float)).all():
        raise ValueError(f"Invalid errors in {path.name}.")

    if (result[["mae", "rmse"]] < 0).any().any():
        raise ValueError(f"Negative errors in {path.name}.")

    validate_horizons(result, ["model", "metric"])
    return result


def read_timed_csv(path, start, end):
    frame = pd.read_csv(path)
    frame["hour_utc"] = pd.to_datetime(
        frame["hour_utc"], utc=True, errors="raise"
    )
    frame = frame.set_index("hour_utc").sort_index()

    expected = pd.date_range(
        start, end, inclusive="left", freq="h", tz="UTC"
    )
    if not frame.index.equals(expected):
        raise ValueError(f"Unexpected timestamps in {path.name}.")

    return frame


def precipitation_baselines():
    train = read_timed_csv(
        DATA_DIR / "splits" / "train.csv",
        "2025-11-01",
        "2026-08-01",
    )
    validation = read_timed_csv(
        DATA_DIR / "supervised" / "validation_y.csv",
        "2026-08-01",
        "2026-09-30",
    )

    training_values = train["precipitation_mm"].to_numpy(float)
    if (
        not np.isfinite(training_values).all()
        or (training_values < 0).any()
    ):
        raise ValueError("Invalid training precipitation.")

    training_mean = float(training_values.mean())
    rows = []

    for horizon in range(1, 25):
        column = f"precipitation_mm_t_plus_{horizon}h"
        actual = validation[column].to_numpy(float)

        if not np.isfinite(actual).all() or (actual < 0).any():
            raise ValueError(f"Invalid precipitation target: {column}")

        for model, constant in (
            ("zero_precipitation", 0.0),
            ("training_mean_precipitation", training_mean),
        ):
            error = actual - constant
            sse = float(np.sum(error**2))
            sst = float(np.sum((actual - actual.mean()) ** 2))

            rows.append(
                {
                    "model": model,
                    "metric": "precipitation_mm",
                    "horizon_hours": horizon,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "r2": 1 - sse / sst if sst > 0 else np.nan,
                    "source_file": (
                        "Computed from training observations "
                        "and validation targets"
                    ),
                }
            )

    return pd.DataFrame(rows), training_mean


def main():
    parts = [
        read_scores(
            DATA_DIR / "evaluation" / "baseline_regression_by_horizon.csv",
            "previous_day",
            "mae",
            "rmse",
            "r2",
        ),
        read_scores(
            MODEL_DIR / RIDGE_RUN / "selected_comparison.csv",
            "ridge",
            "mae_ridge",
            "rmse_ridge",
            "r2_ridge",
        ),
        read_scores(
            MODEL_DIR / RF_RUN / "validation_comparison.csv",
            "random_forest",
            "mae_rf",
            "rmse_rf",
            "r2_rf",
        ),
    ]

    for metric in METRICS:
        sarimax_path = (
            MODEL_DIR
            / SARIMAX_RUNS[metric]
            / metric
            / "baseline_comparison.csv"
        )
        sarimax = read_scores(
            sarimax_path, "sarimax", "mae", "rmse_sarimax", "r2"
        )

        lstm_folder = (
            MODEL_DIR / LSTM_TEMPERATURE_RUN
            if metric == "temperature_c"
            else MODEL_DIR / LSTM_RUN / metric
        )
        lstm = read_scores(
            lstm_folder / "baseline_comparison.csv",
            "lstm",
            "mae_lstm",
            "rmse_lstm",
            "r2_lstm",
        )

        for frame in (sarimax, lstm):
            if set(frame["metric"]) != {metric}:
                raise ValueError(f"Wrong metric in results for {metric}.")

        parts.extend([sarimax, lstm])

    scores = pd.concat(parts, ignore_index=True)
    validate_horizons(scores, ["model", "metric"])

    expected_pairs = {
        (model, metric)
        for model in (
            "previous_day",
            "ridge",
            "random_forest",
            "sarimax",
            "lstm",
        )
        for metric in METRICS
    }
    actual_pairs = set(zip(scores["model"], scores["metric"]))

    if actual_pairs != expected_pairs:
        raise ValueError("Missing or unexpected regression model results.")

    extra_scores, training_mean = precipitation_baselines()
    scores = pd.concat([scores, extra_scores], ignore_index=True)

    means = (
        scores.groupby(["metric", "model"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"))
    )

    comparison = means.pivot(
        index="metric", columns="model", values="mean_rmse"
    ).reindex(METRICS)
    comparison.columns.name = None

    # Selection uses full precision, before rounding for display.
    winners = (
        means.sort_values(["metric", "mean_rmse", "model"])
        .drop_duplicates("metric")
        .set_index("metric")
        .reindex(METRICS)
        .reset_index()
        .rename(columns={"model": "provisional_choice"})
    )

    rain = pd.read_csv(
        MODEL_DIR / RAIN_RUN / "scores_by_horizon.csv"
    )
    validate_horizons(rain, ["model"])

    if (
        not np.isfinite(rain["brier_score"]).all()
        or not rain["brier_score"].between(0, 1).all()
    ):
        raise ValueError("Invalid rain Brier scores.")

    rain_summary = (
        rain.groupby("model", as_index=False)["brier_score"]
        .mean()
        .rename(columns={"brier_score": "mean_brier_score"})
        .sort_values(["mean_brier_score", "model"])
    )

    wind = pd.read_csv(
        MODEL_DIR / WIND_RUN / "scores_by_horizon.csv"
    )
    validate_horizons(wind, ["model"])

    wind_errors = wind[["angular_mae_deg", "angular_rmse_deg"]]
    if (
        not np.isfinite(wind_errors.to_numpy(float)).all()
        or (wind_errors < 0).any().any()
    ):
        raise ValueError("Invalid wind-direction errors.")

    evaluated_counts = wind.pivot(
        index="horizon_hours", columns="model", values="evaluated_rows"
    )
    if evaluated_counts.nunique(axis=1).gt(1).any():
        raise ValueError("Wind models have different evaluation counts.")

    wind_summary = (
        wind.groupby("model", as_index=False)
        .agg(
            mean_angular_mae_deg=("angular_mae_deg", "mean"),
            mean_angular_rmse_deg=("angular_rmse_deg", "mean"),
        )
        .sort_values(["mean_angular_mae_deg", "model"])
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output = DATA_DIR / "evaluation" / f"model_comparison_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    scores.to_csv(output / "regression_by_horizon.csv", index=False)
    comparison.to_csv(output / "regression_mean_rmse.csv")
    winners.to_csv(output / "provisional_regression_choices.csv", index=False)
    rain_summary.to_csv(output / "rain_probability_comparison.csv", index=False)
    wind_summary.to_csv(output / "wind_direction_comparison.csv", index=False)

    print("\nMean validation RMSE across 24 horizons:")
    print(comparison.round(6).to_string(na_rep="—"))

    print("\nProvisional regression choices:")
    print(winners.round(6).to_string(index=False))

    print(
        "\nTraining-average precipitation baseline: "
        f"{training_mean:.6f} mm/hour"
    )
    print("\nPrecipitation comparison — lower RMSE is better:")
    print(
        means.loc[means["metric"].eq("precipitation_mm")]
        .sort_values("mean_rmse")
        .round(6)
        .to_string(index=False)
    )

    print("\nRain probability — lower Brier score is better:")
    print(rain_summary.round(6).to_string(index=False))

    print("\nWind direction — lower angular MAE is better:")
    print(wind_summary.round(6).to_string(index=False))

    print(f"\nSaved comparison reports to: {output}")
    print("Existing model scores were consolidated, not independently audited.")
    print("Precipitation constant baselines were computed from train/validation.")
    print("Rain alert thresholds and probability calibration remain unresolved.")
    print("Choices are provisional validation results, not final test results.")
    print("No models retrained or replaced.")
    print("Test data not read. No Hopsworks upload.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Comparison failed: {error}")
        raise SystemExit(1) from None