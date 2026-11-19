from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "splits" / "train.csv"
OUTPUT_DIR = BASE_DIR / "data" / "eda" / "training"

LABELS = {
    "temperature_c": "Temperature (°C)",
    "humidity_pct": "Humidity (%)",
    "precipitation_mm": "Precipitation (mm/hour)",
    "rain_mm": "Rain (mm/hour)",
    "wind_speed_ms": "Wind speed (m/s)",
    "wind_direction_deg": "Wind direction (degrees)",
    "us_aqi": "US AQI",
    "pm2_5_ug_m3": "PM2.5 (µg/m³)",
    "pm10_ug_m3": "PM10 (µg/m³)",
    "uv_index": "UV Index",
}


def save_chart(fig, filename):
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved chart: {filename}")


def load_training_data():
    df = pd.read_csv(INPUT_PATH)

    required = {"hour_utc", *LABELS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["hour_utc"] = pd.to_datetime(
        df["hour_utc"], utc=True, errors="raise"
    )
    df = df.sort_values("hour_utc").set_index("hour_utc")

    expected = pd.date_range(
        "2025-11-01",
        "2026-08-01",
        inclusive="left",
        freq="h",
        tz="UTC",
    )
    if not df.index.equals(expected):
        raise ValueError("Training timestamps do not match the expected period.")

    for column in LABELS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(df[list(LABELS)].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite metric values found.")

    return df


def make_distributions(df):
    fig, axes = plt.subplots(5, 2, figsize=(13, 17), layout="constrained")

    for ax, (column, label) in zip(
        axes.flat, LABELS.items(), strict=True
    ):
        ax.hist(df[column], bins=40, color="#167D9A", edgecolor="white")
        ax.set_title(label)
        ax.set_ylabel("Hourly records")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "AgentWeather — Training distributions\n"
        "ERA5 weather and CAMS modelled air quality",
        fontsize=15,
    )
    save_chart(fig, "01_distributions.png")


def make_daily_trends(df):
    aggregations = {
        "temperature_c": "mean",
        "humidity_pct": "mean",
        "precipitation_mm": "sum",
        "wind_speed_ms": "mean",
        "us_aqi": "mean",
        "uv_index": "max",
    }

    daily = df[list(aggregations)].resample("D").agg(aggregations)
    daily.to_csv(OUTPUT_DIR / "daily_summary_utc.csv")

    titles = [
        "Daily mean temperature (°C)",
        "Daily mean humidity (%)",
        "Daily total precipitation (mm)",
        "Daily mean wind speed (m/s)",
        "Daily mean modelled US AQI",
        "Daily maximum UV Index",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")

    for ax, column, title in zip(
        axes.flat, aggregations, titles, strict=True
    ):
        ax.plot(daily.index, daily[column], color="#167D9A", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)

    fig.suptitle("AgentWeather — Training trends (UTC days)", fontsize=15)
    save_chart(fig, "02_daily_trends.png")


def make_hourly_patterns(df):
    columns = ["temperature_c", "humidity_pct", "us_aqi", "uv_index"]
    local_hours = df.index.tz_convert("Asia/Karachi").hour
    hourly = df[columns].groupby(local_hours).mean()
    hourly.index.name = "hour_pakistan_time"
    hourly.to_csv(OUTPUT_DIR / "hourly_patterns_pkt.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

    for ax, column in zip(axes.flat, columns, strict=True):
        ax.plot(hourly.index, hourly[column], marker="o", color="#167D9A")
        ax.set_title(f"Mean {LABELS[column]}")
        ax.set_xlabel("Hour in Karachi (PKT)")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.2)

    fig.suptitle("Average daily patterns — training period", fontsize=15)
    save_chart(fig, "03_hourly_patterns.png")


def summarize_rain(df):
    # Exploratory thresholds, not a final rain-event definition.
    rain = df["rain_mm"]
    summary = pd.DataFrame(
        [
            {
                "definition": "rain_mm > 0",
                "hours": int(rain.gt(0).sum()),
                "percent": float(rain.gt(0).mean() * 100),
            },
            {
                "definition": "rain_mm >= 0.1",
                "hours": int(rain.ge(0.1).sum()),
                "percent": float(rain.ge(0.1).mean() * 100),
            },
            {
                "definition": "rain_mm >= 1.0",
                "hours": int(rain.ge(1.0).sum()),
                "percent": float(rain.ge(1.0).mean() * 100),
            },
        ]
    )
    summary.to_csv(OUTPUT_DIR / "rain_frequency.csv", index=False)

    print("\nRain frequency — exploratory thresholds:")
    print(summary.round(2).to_string(index=False))
    print("These percentages are not forecast rain probabilities.")


def main():
    try:
        df = load_training_data()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Direction is circular: do not report an ordinary mean for it.
        scalar_columns = [
            column for column in LABELS
            if column != "wind_direction_deg"
        ]

        df[scalar_columns].describe(
            percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
        ).T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

        make_distributions(df)
        make_daily_trends(df)
        make_hourly_patterns(df)
        summarize_rain(df)

        # Keep unusual values visible for review; do not delete them.
        df.nlargest(10, "pm10_ug_m3")[
            ["pm10_ug_m3", "pm2_5_ug_m3", "us_aqi"]
        ].to_csv(OUTPUT_DIR / "highest_pm10_hours.csv")

        print(f"\nTraining rows analysed: {len(df)}")
        print(f"Charts and tables saved to: {OUTPUT_DIR}")
        print("Validation and test files were not read.")
        print("No source data changed and no Hopsworks upload.")

    except (ValueError, KeyError, TypeError) as error:
        print(f"EDA failed: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("File access failed. Check paths and close open output files.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()