import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

from evaluate_baseline import regression_scores

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

METRICS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
]

BOUNDS = {
    "us_aqi": (0, None),
    "temperature_c": (None, None),
    "precipitation_mm": (0, None),
    "humidity_pct": (0, 100),
    "uv_index": (0, None),
    "wind_speed_ms": (0, None),
}

# Initial fixed candidate: hourly dependence plus daily seasonality.
ORDER = (1, 0, 0)
SEASONAL_ORDER = (1, 0, 0, 24)


def load_frame(path, start, end):
    frame = pd.read_csv(path)

    if "hour_utc" not in frame.columns:
        raise ValueError(f"Missing hour_utc in {path.name}.")

    frame["hour_utc"] = pd.to_datetime(
        frame["hour_utc"], utc=True, errors="raise"
    )
    frame = frame.set_index("hour_utc").sort_index()

    expected = pd.date_range(
        start=start,
        end=end,
        inclusive="left",
        freq="h",
        tz="UTC",
        name="hour_utc",
    )

    if not frame.index.equals(expected):
        raise ValueError(
            f"Unexpected timestamps or duplicate hours in {path.name}."
        )

    return frame


def load_inputs():
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
    targets = load_frame(
        DATA_DIR / "supervised" / "validation_y.csv",
        "2026-08-01",
        "2026-09-30",
    )

    baseline = pd.read_csv(
        DATA_DIR / "evaluation" / "baseline_regression_by_horizon.csv"
    )

    return train, validation, targets, baseline


def validate_metric(metric, train, validation, targets, baseline):
    for name, frame in (
        ("training", train),
        ("validation", validation),
    ):
        values = frame[metric].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Invalid {metric} values in {name}.")

    columns = [
        f"{metric}_t_plus_{horizon}h"
        for horizon in range(1, 25)
    ]

    if not np.isfinite(targets[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid targets for {metric}.")

    # Confirm every label represents exactly t + horizon.
    for horizon, column in enumerate(columns, start=1):
        future_times = targets.index + pd.Timedelta(hours=horizon)

        expected = validation.loc[
            future_times, metric
        ].to_numpy(dtype=float)

        actual = targets[column].to_numpy(dtype=float)

        if not np.allclose(actual, expected, rtol=0, atol=1e-8):
            raise ValueError(f"Target alignment failed: {column}.")

    reference = baseline.loc[
        baseline["metric"].eq(metric),
        ["metric", "horizon_hours", "rmse"],
    ].copy()

    if (
        len(reference) != 24
        or reference["horizon_hours"].duplicated().any()
        or set(reference["horizon_hours"]) != set(range(1, 25))
    ):
        raise ValueError(f"Incomplete baseline results for {metric}.")

    rmse = reference["rmse"].to_numpy(dtype=float)

    if not np.isfinite(rmse).all() or (rmse < 0).any():
        raise ValueError(f"Invalid baseline RMSE for {metric}.")

    return columns, reference


def train_metric(metric, train, validation, targets, baseline):
    columns, reference = validate_metric(
        metric, train, validation, targets, baseline
    )

    values = train[metric].to_numpy(dtype=float)

    # Learn normalization only from the training period.
    mean = float(values.mean())
    scale = float(values.std(ddof=0))

    if scale <= 0:
        raise ValueError(f"Training series is constant: {metric}.")

    standardized = (values - mean) / scale

    print(f"\nFitting SARIMAX: {metric}", flush=True)
    print(f"Order: {ORDER}; seasonal order: {SEASONAL_ORDER}", flush=True)

    fitted = SARIMAX(
        standardized,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(
        disp=False,
        maxiter=200,
    )

    if not fitted.mle_retvals.get("converged", False):
        raise ValueError(
            f"{metric}: optimizer did not converge. "
            "This candidate has not been accepted."
        )

    parameters = np.asarray(fitted.params, dtype=float)

    if not np.isfinite(parameters).all():
        raise ValueError(f"Non-finite fitted parameters for {metric}.")

    state = fitted
    prediction_values = np.empty((len(targets), 24), dtype=float)
    previous_time = train.index[-1]

    print(
        f"Evaluating {len(targets)} rolling prediction times...",
        flush=True,
    )

    for position, prediction_time in enumerate(targets.index):
        if prediction_time != previous_time + pd.Timedelta(hours=1):
            raise ValueError("A gap occurred in the rolling evaluation.")

        # At time t, the observation at t is available.
        # No observations after t are supplied to the model.
        observed = float(validation.at[prediction_time, metric])
        state = state.extend(
            np.array([(observed - mean) / scale], dtype=float)
        )

        # Forecast t+1 through t+24.
        forecast = (
            np.asarray(state.forecast(steps=24), dtype=float)
            .reshape(-1)
            * scale
            + mean
        )

        if len(forecast) != 24 or not np.isfinite(forecast).all():
            raise ValueError(
                f"Invalid forecast for {metric} at {prediction_time}."
            )

        prediction_values[position] = forecast
        previous_time = prediction_time

        completed = position + 1
        if completed % 120 == 0 or completed == len(targets):
            print(
                f"  {metric}: {completed}/{len(targets)}",
                flush=True,
            )

    raw = pd.DataFrame(
        prediction_values,
        index=targets.index,
        columns=columns,
    )

    lower, upper = BOUNDS[metric]
    bounded = raw.clip(lower=lower, upper=upper)

    score_rows = []

    for variant, predictions in (("raw", raw), ("bounded", bounded)):
        for horizon, column in enumerate(columns, start=1):
            score_rows.append(
                {
                    "metric": metric,
                    "variant": variant,
                    "horizon_hours": horizon,
                    **regression_scores(
                        targets[column].to_numpy(dtype=float),
                        predictions[column].to_numpy(dtype=float),
                    ),
                }
            )

    scores = pd.DataFrame(score_rows)

    comparison = (
        scores.loc[scores["variant"].eq("bounded")]
        .drop(columns="variant")
        .merge(
            reference,
            on=["metric", "horizon_hours"],
            validate="one_to_one",
            suffixes=("_sarimax", "_baseline"),
        )
    )

    metadata = {
        "metric": metric,
        "order": list(ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "trend": "c",
        "enforce_stationarity": True,
        "enforce_invertibility": True,
        "training_mean": mean,
        "training_scale": scale,
        "parameter_names": list(fitted.param_names),
        "parameters": parameters.tolist(),
        "converged": True,
        "training_start": train.index.min().isoformat(),
        "training_end": train.index.max().isoformat(),
        "validation_origin_start": targets.index.min().isoformat(),
        "validation_origin_end": targets.index.max().isoformat(),
        "bounds": list(BOUNDS[metric]),
        "statsmodels_version": statsmodels.__version__,
        "evaluation": (
            "Observe time t, update state without parameter refitting, "
            "then forecast t+1 through t+24."
        ),
        "availability_assumption": (
            "Historical observation at time t is available at time t. "
            "Actual source publication delays are not simulated."
        ),
        "artifact_note": (
            "Parameters and specification only; deployment also requires "
            "reconstructing the filtered state from historical observations."
        ),
    }

    return raw, bounded, scores, comparison, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train and validate statistical forecasting candidates."
    )
    parser.add_argument(
        "--metric",
        choices=[*METRICS, "all"],
        default="temperature_c",
    )
    args = parser.parse_args()

    try:
        train, validation, targets, baseline = load_inputs()

        metrics = METRICS if args.metric == "all" else [args.metric]

        print(f"Training observations: {len(train)}")
        print(f"Validation prediction times: {len(targets)}")
        print("Forecast horizons: 1–24 hours")
        print("Parameters are fitted on training data only.")

        run_dir = None
        summaries = []

        for metric in metrics:
            raw, bounded, scores, comparison, metadata = train_metric(
                metric, train, validation, targets, baseline
            )

            if run_dir is None:
                run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
                run_dir = DATA_DIR / "models" / f"sarimax_{run_id}"
                run_dir.mkdir(parents=True, exist_ok=False)

            metric_dir = run_dir / metric
            metric_dir.mkdir()

            raw.to_csv(
                metric_dir / "raw_validation_predictions.csv",
                index_label="hour_utc",
            )
            bounded.to_csv(
                metric_dir / "validation_predictions.csv",
                index_label="hour_utc",
            )
            scores.to_csv(
                metric_dir / "raw_and_bounded_scores.csv",
                index=False,
            )
            comparison.to_csv(
                metric_dir / "baseline_comparison.csv",
                index=False,
            )
            with (metric_dir / "model_specification.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(metadata, file, indent=2, allow_nan=False)

            sarimax_rmse = float(comparison["rmse_sarimax"].mean())
            baseline_rmse = float(comparison["rmse_baseline"].mean())

            summaries.append(
                {
                    "metric": metric,
                    "sarimax_mean_rmse": sarimax_rmse,
                    "baseline_mean_rmse": baseline_rmse,
                    "sarimax_better": sarimax_rmse < baseline_rmse,
                }
            )

            print("\nBounded SARIMAX results at +24 hours:")
            print(
                comparison.loc[
                    comparison["horizon_hours"].eq(24)
                ].round(4).to_string(index=False)
            )
            print(f"Saved metric results to: {metric_dir}")

            # Save progress after each successfully completed metric.
            pd.DataFrame(summaries).to_csv(
                run_dir / "summary.csv",
                index=False,
            )

        print("\nMean validation RMSE across all 24 horizons:")
        print(pd.DataFrame(summaries).round(4).to_string(index=False))

        print(f"\nResults saved to: {run_dir}")
        print("Candidate parameters saved; no production model replaced.")
        print("Test data not read. No Hopsworks upload.")
        return 0

    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        print(f"\nStatistical training failed: {error}")
        return 1
    except OSError as error:
        print(f"\nFile access failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())