from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "historical_hourly_2025-11_2026-10.csv"
)
TARGET_PATH = BASE_DIR / "data" / "supervised" / "validation_y.csv"
OUTPUT_DIR = BASE_DIR / "data" / "evaluation"

METRICS = [
    "us_aqi",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_ms",
]


def read_indexed(path):
    df = pd.read_csv(path)
    df["hour_utc"] = pd.to_datetime(
        df["hour_utc"], utc=True, errors="raise"
    )

    if df["hour_utc"].isna().any() or df["hour_utc"].duplicated().any():
        raise ValueError(f"Invalid timestamps in {path.name}.")

    return df.set_index("hour_utc").sort_index()


def regression_scores(actual, predicted):
    error = predicted - actual
    denominator = np.sum((actual - actual.mean()) ** 2)

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": (
            float(1 - np.sum(error**2) / denominator)
            if denominator > 0
            else np.nan
        ),
    }


def main():
    try:
        history = read_indexed(HISTORY_PATH)

        # Exclude test-period values immediately.
        history = history.loc[
            history.index < pd.Timestamp("2026-10-01", tz="UTC")
        ]
        targets = read_indexed(TARGET_PATH)

        expected_origins = pd.date_range(
            "2026-08-01",
            "2026-09-30",
            inclusive="left",
            freq="h",
            tz="UTC",
        )
        if not targets.index.equals(expected_origins):
            raise ValueError("Unexpected validation prediction times.")

        required_history = [
            *METRICS,
            "rain_mm",
            "wind_direction_deg",
        ]
        for column in required_history:
            history[column] = pd.to_numeric(
                history[column], errors="raise"
            )

        if not np.isfinite(
            history[required_history].to_numpy(dtype=float)
        ).all():
            raise ValueError("Missing or invalid history values.")

        regression_rows = []
        rain_rows = []
        direction_rows = []

        for horizon in range(1, 25):
            # Forecast for t+h uses the historical value at t+h-24.
            reference_times = targets.index + pd.Timedelta(
                hours=horizon - 24
            )
            future_times = targets.index + pd.Timedelta(hours=horizon)

            if not (reference_times <= targets.index).all():
                raise ValueError("Baseline uses information from the future.")

            reference = history.reindex(reference_times)
            future = history.reindex(future_times)

            if reference[required_history].isna().any().any():
                raise ValueError("Missing baseline reference values.")

            if future[required_history].isna().any().any():
                raise ValueError("Missing validation outcome values.")

            for metric in METRICS:
                actual = targets[
                    f"{metric}_t_plus_{horizon}h"
                ].to_numpy(dtype=float)
                predicted = reference[metric].to_numpy(dtype=float)

                if not np.isfinite(actual).all():
                    raise ValueError(f"Invalid targets for {metric}.")

                if not np.allclose(
                    actual,
                    future[metric].to_numpy(dtype=float),
                    rtol=0,
                    atol=1e-9,
                ):
                    raise ValueError(f"Target alignment failed for {metric}.")

                regression_rows.append(
                    {
                        "metric": metric,
                        "horizon_hours": horizon,
                        **regression_scores(actual, predicted),
                    }
                )

            # Previous-day rain occurrence gives a deterministic 0/1 forecast.
            actual_rain = targets[
                f"rain_event_t_plus_{horizon}h"
            ].to_numpy(dtype=float)

            if not np.isin(actual_rain, [0, 1]).all():
                raise ValueError("Rain targets must be zero or one.")

            expected_rain = (
                future["rain_mm"].to_numpy(dtype=float) >= 0.1
            ).astype(int)

            if not np.array_equal(actual_rain, expected_rain):
                raise ValueError("Rain target alignment failed.")

            predicted_rain = (
                reference["rain_mm"].to_numpy(dtype=float) >= 0.1
            ).astype(int)

            actual_event = actual_rain.astype(bool)
            predicted_event = predicted_rain.astype(bool)

            tp = int(np.sum(actual_event & predicted_event))
            fp = int(np.sum(~actual_event & predicted_event))
            fn = int(np.sum(actual_event & ~predicted_event))

            rain_rows.append(
                {
                    "horizon_hours": horizon,
                    "actual_rain_hours": int(actual_event.sum()),
                    "predicted_rain_hours": int(predicted_event.sum()),
                    "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "recall": tp / (tp + fn) if tp + fn else np.nan,
                    "f1": (
                        2 * tp / (2 * tp + fp + fn)
                        if 2 * tp + fp + fn
                        else np.nan
                    ),
                    "brier_score": float(
                        np.mean((predicted_rain - actual_rain) ** 2)
                    ),
                    "always_dry_brier": float(np.mean(actual_rain**2)),
                }
            )

            # Angular error wraps at north: 359° vs 1° is a 2° error.
            # A previous-day direction forecast is unavailable if it was calm.
            valid = (
                future["wind_speed_ms"].to_numpy(dtype=float) > 0
            ) & (
                reference["wind_speed_ms"].to_numpy(dtype=float) > 0
            )

            actual_direction = future[
                "wind_direction_deg"
            ].to_numpy(dtype=float)
            predicted_direction = reference[
                "wind_direction_deg"
            ].to_numpy(dtype=float)

            angular_error = (
                (predicted_direction - actual_direction + 180) % 360
            ) - 180
            errors = angular_error[valid]

            direction_rows.append(
                {
                    "horizon_hours": horizon,
                    "evaluated_rows": int(valid.sum()),
                    "excluded_calm_rows": int((~valid).sum()),
                    "angular_mae_deg": (
                        float(np.abs(errors).mean())
                        if len(errors)
                        else np.nan
                    ),
                    "angular_rmse_deg": (
                        float(np.sqrt(np.mean(errors**2)))
                        if len(errors)
                        else np.nan
                    ),
                }
            )

        regression = pd.DataFrame(regression_rows)
        rain = pd.DataFrame(rain_rows)
        direction = pd.DataFrame(direction_rows)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        regression.to_csv(
            OUTPUT_DIR / "baseline_regression_by_horizon.csv", index=False
        )
        rain.to_csv(
            OUTPUT_DIR / "baseline_rain_by_horizon.csv", index=False
        )
        direction.to_csv(
            OUTPUT_DIR / "baseline_direction_by_horizon.csv", index=False
        )

        print("\nPrevious-day baseline — validation")
        print(f"Prediction times: {len(targets)}")
        print("Forecast horizons: 1–24 hours")

        print("\nRegression results at +24 hours:")
        print(
            regression.loc[
                regression["horizon_hours"].eq(24),
                ["metric", "mae", "rmse", "r2"],
            ].round(4).to_string(index=False)
        )

        print("\nRain-event results at +24 hours:")
        print(
            rain.loc[rain["horizon_hours"].eq(24)]
            .round(4)
            .to_string(index=False)
        )

        print("\nWind-direction results at +24 hours:")
        print(
            direction.loc[direction["horizon_hours"].eq(24)]
            .round(4)
            .to_string(index=False)
        )

        print(f"\nAll horizon results saved to: {OUTPUT_DIR}")
        print("Undefined scores are stored as blank values.")
        print("No model trained. Test targets were not read.")
        print("No Hopsworks upload.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"Baseline evaluation failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and close open output files.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()