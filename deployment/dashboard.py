"""Display saved historical and experimental AgentWeather results."""

import json
import re
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from report_agent import read_forecast

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "demo_data"
TEST_DIR = DATA_DIR / "evaluation/final_test_20261121T112620710620"
EDA_DIR = DATA_DIR / "eda/training"
EXPERIMENT_DIR = DATA_DIR / "experimental_forecasts"
CLOUD_DIR = BASE_DIR / "cloud_data" / "experimental_forecasts"

HISTORICAL_MODE = "retrospective_historical_demo"
EXPERIMENTAL_MODE = "experimental_mixed_source"

METRICS = {
    "Temperature": ("temperature_c", "Temperature (°C)"),
    "Humidity": ("humidity_pct", "Humidity (%)"),
    "Air quality": ("us_aqi", "Modelled US AQI"),
    "UV Index": ("uv_index", "UV Index"),
    "Wind speed": ("wind_speed_ms", "Wind speed (m/s)"),
    "Precipitation": ("precipitation_mm", "Hourly precipitation (mm)"),
    "Rain probability": (
        "rain_probability",
        "Hourly rain probability (%)",
    ),
}

MODEL_NAMES = {
    "selected_models": "Selected models",
    "random_forest": "Random Forest",
    "lstm": "LSTM",
    "sarimax": "SARIMAX",
    "ridge": "Ridge",
    "persistence": "Latest-value baseline",
    "previous_day": "Previous-day baseline",
    "zero_precipitation": "Zero-rainfall baseline",
    "training_mean_precipitation": "Training-average baseline",
    "always_dry": "Always-dry baseline",
    "training_frequency": "Training-frequency baseline",
    "logistic": "Logistic regression",
}

ERRORS = (OSError, ValueError, KeyError, TypeError)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path.name}")
    return value


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def format_time(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Timestamp must contain a timezone.")
    return timestamp.tz_convert("Asia/Karachi").strftime(
        "%d %b %Y, %H:%M PKT"
    )


def prepare_display(frame):
    frame = frame.copy()
    frame["valid_time_utc"] = pd.to_datetime(
        frame["valid_time_utc"], utc=True, errors="raise"
    )
    frame = frame.sort_values("valid_time_utc")
    frame["Time (PKT)"] = (
        frame["valid_time_utc"]
        .dt.tz_convert("Asia/Karachi")
        .dt.tz_localize(None)
    )
    return frame


def load_historical():
    for path in sorted(
        (DATA_DIR / "agent_logs").glob("*.json"), reverse=True
    ):
        try:
            log = read_json(path)
        except ERRORS:
            continue

        if (
            log.get("status") != "success"
            or log.get("mode") != HISTORICAL_MODE
            or not log.get("monitoring_path")
            or not log.get("report_path")
        ):
            continue

        run_id = log["run_id"]
        if (
            not isinstance(run_id, str)
            or not re.fullmatch(r"\d{8}T\d{12}_[a-f0-9]{8}", run_id)
            or path.stem != run_id
        ):
            raise ValueError("Invalid historical run identifier.")

        forecast_path = (
            DATA_DIR / "forecasts" / f"agent_forecast_{run_id}.csv"
        )
        rows, origin, forecast_run = read_forecast(forecast_path)

        if forecast_run != run_id:
            raise ValueError("Historical forecast and log do not match.")
        if pd.Timestamp(log["origin_utc"]) != pd.Timestamp(origin):
            raise ValueError("Historical origins do not match.")

        alerts = read_json(
            DATA_DIR / "alerts" / f"agent_alerts_{run_id}.json"
        )
        monitoring = read_json(
            DATA_DIR / "monitoring" / f"agent_monitoring_{run_id}.json"
        )
        for document in (alerts, monitoring):
            if (
                document.get("agent_run_id") != run_id
                or document.get("mode") != HISTORICAL_MODE
            ):
                raise ValueError("Historical outputs belong to different runs.")

        if pd.Timestamp(alerts["prediction_time_utc"]) != pd.Timestamp(origin):
            raise ValueError("Alert origin differs.")
        if pd.Timestamp(monitoring["window_end_utc"]) != pd.Timestamp(origin):
            raise ValueError("Monitoring origin differs.")

        report = (
            DATA_DIR / "reports" / f"agent_report_{run_id}.md"
        ).read_text(encoding="utf-8")
        if run_id not in report:
            raise ValueError("Historical report identifier differs.")

        return log, prepare_display(pd.DataFrame(rows)), alerts, monitoring, report

    raise FileNotFoundError("No completed historical workflow was found.")


def list_experimental_runs():
    runs = []
    paths = [
        *EXPERIMENT_DIR.glob("*/execution_log.json"),
        *CLOUD_DIR.glob("*/execution_log.json"),
    ]
    for path in sorted(paths, key=lambda item: item.parent.name, reverse=True):
        try:
            log = read_json(path)
            if (
                log.get("status") == "success"
                and log.get("mode") == EXPERIMENTAL_MODE
                and log.get("run_id") == path.parent.name
                and (path.parent / "forecast.csv").is_file()
                and (path.parent / "baseline_predictions.csv").is_file()
            ):
                runs.append(path.parent)
        except ERRORS:
            continue
    return runs


def load_experimental(run_dir):
    if run_dir.parent == CLOUD_DIR:
        publication = read_json(CLOUD_DIR.parent / "publication.json")
        if publication["run_id"] != run_dir.name:
            raise ValueError("Cloud publication run ID differs.")
        for name in ("forecast.csv", "baseline_predictions.csv", "execution_log.json"):
            if file_hash(run_dir / name) != publication["files"][name]:
                raise ValueError(f"Cloud publication checksum mismatch: {name}")
    log = read_json(run_dir / "execution_log.json")
    frame = pd.read_csv(run_dir / "forecast.csv")

    if not frame["forecast_mode"].eq(EXPERIMENTAL_MODE).all():
        raise ValueError("Forecast is not marked experimental.")

    frame = frame.sort_values("horizon_hours")
    if frame["horizon_hours"].tolist() != list(range(1, 25)):
        raise ValueError("Expected exactly 24 forecast horizons.")

    for column in (
        "prediction_time_utc",
        "valid_time_utc",
        "generated_at_utc",
    ):
        frame[column] = pd.to_datetime(
            frame[column], utc=True, errors="raise"
        )
        if frame[column].isna().any():
            raise ValueError(f"Missing timestamp: {column}")

    if (
        frame["prediction_time_utc"].nunique() != 1
        or frame["generated_at_utc"].nunique() != 1
    ):
        raise ValueError("Expected one cutoff and generation time.")

    origin = frame["prediction_time_utc"].iloc[0]
    generated = frame["generated_at_utc"].iloc[0]

    if (
        origin != pd.Timestamp(log["input_cutoff_utc"])
        or generated != pd.Timestamp(log["generated_at_utc"])
    ):
        raise ValueError("Forecast and execution log timestamps differ.")

    expected = origin + pd.to_timedelta(
        frame["horizon_hours"].to_numpy(), unit="h"
    )
    if not pd.DatetimeIndex(frame["valid_time_utc"]).equals(
        pd.DatetimeIndex(expected)
    ):
        raise ValueError("Incorrect forecast target timestamps.")

    columns = [details[0] for details in METRICS.values()]
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ValueError("Missing or invalid forecast values.")

    if not frame["rain_probability"].between(0, 1).all():
        raise ValueError("Invalid rain probabilities.")
    if not frame["humidity_pct"].between(0, 100).all():
        raise ValueError("Invalid humidity predictions.")

    nonnegative = [
        "us_aqi", "uv_index", "precipitation_mm", "wind_speed_ms"
    ]
    if frame[nonnegative].lt(0).any().any():
        raise ValueError("Unexpected negative predictions.")

    return log, prepare_display(frame)


def load_evaluation(run_dir):
    """Select the latest completed check bound to these forecast files."""
    hashes = {
        name: file_hash(run_dir / name)
        for name in ("forecast.csv", "baseline_predictions.csv")
    }

    candidates = []
    for path in (run_dir / "evaluations").glob(
        "*/evaluation_metadata.json"
    ):
        try:
            metadata = read_json(path)
            if metadata.get("input_hashes") != hashes:
                continue

            folder = path.parent
            if not (folder / "timing.csv").is_file():
                continue

            checked = pd.Timestamp(metadata["checked_at_utc"])
            if checked.tzinfo is None:
                continue

            # A no-API pending check is complete only when no eligible
            # interval had ended at its recorded check time.
            generated = pd.Timestamp(
                metadata["forecast_generated_at_utc"]
            )
            timing = pd.read_csv(folder / "timing.csv")
            targets = pd.to_datetime(
                timing["valid_time_utc"], utc=True, errors="raise"
            )
            eligible = (
                targets.sub(pd.Timedelta(hours=1)).ge(generated)
                & targets.le(checked)
            )
            scores_path = folder / "scores.csv"

            if eligible.any() and not scores_path.is_file():
                continue

            scores = (
                pd.read_csv(scores_path) if scores_path.is_file() else None
            )
            candidates.append((checked, metadata, scores))
        except ERRORS:
            continue

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])


def style_chart(figure, label):
    figure.update_layout(
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#172B4D", "size": 13},
        margin={"l": 20, "r": 20, "t": 25, "b": 25},
        yaxis_title=label,
        hovermode="x unified",
        legend_title_text="",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#E8EEF5", zeroline=False)
    return figure


def overview(frame):
    st.subheader("Karachi forecast overview")
    st.caption("Cards summarize the saved 24-hour prediction period.")

    cards = st.columns(3)
    cards[0].metric(
        "Temperature range",
        f"{frame['temperature_c'].min():.1f}–"
        f"{frame['temperature_c'].max():.1f} °C",
    )
    cards[1].metric("Highest modelled AQI", f"{frame['us_aqi'].max():.1f}")
    cards[2].metric(
        "Total predicted precipitation",
        f"{frame['precipitation_mm'].sum():.2f} mm",
    )

    cards = st.columns(3)
    cards[0].metric(
        "Humidity range",
        f"{frame['humidity_pct'].min():.0f}–"
        f"{frame['humidity_pct'].max():.0f}%",
    )
    cards[1].metric("Highest UV Index", f"{frame['uv_index'].max():.2f}")
    cards[2].metric(
        "Highest wind speed", f"{frame['wind_speed_ms'].max():.1f} m/s"
    )

    st.divider()
    selected = st.selectbox("Weather metric", list(METRICS))
    column, label = METRICS[selected]
    chart = frame[["Time (PKT)", column]].copy()

    if column == "rain_probability":
        chart[column] *= 100

    arguments = {
        "data_frame": chart,
        "x": "Time (PKT)",
        "y": column,
        "labels": {column: label},
        "color_discrete_sequence": ["#2563EB"],
    }
    figure = (
        px.bar(**arguments)
        if column == "precipitation_mm"
        else px.line(**arguments, markers=True)
    )
    style_chart(figure, label)

    if column in ("humidity_pct", "rain_probability"):
        figure.update_yaxes(range=[0, 100])
    elif column != "temperature_c":
        figure.update_yaxes(rangemode="tozero")

    st.plotly_chart(figure, width="stretch")
    st.caption("Saved predictions, not observed conditions. Times are PKT.")

    if column == "rain_probability":
        st.caption(
            "Probability applies to each individual hour, not the whole day. "
            "Calibration is not established; rain alerts are disabled."
        )
    elif column == "us_aqi":
        st.caption("Air-quality references are model estimates, not station data.")

    with st.expander("View hourly values"):
        st.dataframe(
            chart.rename(columns={column: label}).round(3),
            hide_index=True,
            width="stretch",
        )


def historical_report(alerts, report):
    st.subheader("Historical alert summary")
    st.caption("Saved historical flags, not current warnings.")

    names = {"us_aqi": "Air quality", "uv_index": "UV exposure"}
    for alert in alerts["alerts"]:
        with st.container(border=True):
            st.markdown(
                f"**{names.get(alert['metric'], alert['metric'])}: "
                f"{alert['category_at_peak']}**"
            )
            st.write(
                f"{format_time(alert['first_target_time_utc'])} to "
                f"{format_time(alert['last_target_time_utc'])}"
            )
            st.caption(
                f"{alert['flagged_hours']} flagged hours. "
                f"Peak: {alert['peak_predicted_value']:.2f}. "
                "Category applies to the window peak."
            )

    if not alerts["alerts"]:
        st.write("No saved AQI or UV alert windows.")

    st.caption("Rain alerts are disabled. No notifications were sent.")
    st.divider()
    st.subheader("Weather report")
    st.caption("Factual template based on saved predictions.")

    parts = report.split("## Weather outlook", 1)
    narrative = (
        parts[1].split("## Historical alert evaluation", 1)[0]
        if len(parts) == 2 else report
    )
    st.markdown(narrative)
    st.download_button(
        "Download complete report",
        report,
        file_name="AgentWeather_historical_report.md",
        mime="text/markdown",
    )


def experimental_report(frame):
    st.subheader("Experimental forecast summary")
    st.info("Experimental alerts are disabled. No notifications are issued.")

    st.write(
        f"For the saved prediction period, temperature ranges from "
        f"{frame['temperature_c'].min():.1f} to "
        f"{frame['temperature_c'].max():.1f} °C, with humidity from "
        f"{frame['humidity_pct'].min():.0f}% to "
        f"{frame['humidity_pct'].max():.0f}%."
    )
    st.write(
        f"Predicted modelled AQI ranges from {frame['us_aqi'].min():.1f} "
        f"to {frame['us_aqi'].max():.1f}. "
        f"The highest predicted UV Index is {frame['uv_index'].max():.2f}."
    )
    st.write(
        f"The sum of hourly precipitation predictions is "
        f"{frame['precipitation_mm'].sum():.2f} mm. Individual-hour rain "
        f"probabilities range from "
        f"{100 * frame['rain_probability'].min():.1f}% to "
        f"{100 * frame['rain_probability'].max():.1f}%."
    )
    st.caption(
        "This summary is generated from saved numbers, without an LLM. "
        "Accuracy with GFS inputs and rain-probability calibration are "
        "not established. Wind direction is omitted pending its display policy."
    )


def experimental_performance(run_dir):
    st.subheader("Experimental model performance")
    st.write(
        "Comparison with ERA5 reanalysis and CAMS estimates, "
        "not independent station observations."
    )

    evaluation = load_evaluation(run_dir)
    if evaluation is None:
        st.info("No completed evaluation matching these forecast files was found.")
        return

    checked, _metadata, scores = evaluation
    st.caption(f"Latest saved evaluation: {format_time(checked)}")
    st.caption(
        "This page does not fetch new references. Results reflect the saved "
        "evaluation time, even if more target hours have since passed."
    )

    if scores is None:
        st.info("Pending: no eligible intervals had ended at this check.")
        return

    required = {
        "metric", "method", "scored_hours", "reference_pending_hours",
        "excluded_other_hours", "eligible_elapsed_hours",
        "mae", "rmse", "brier_score",
    }
    if not required.issubset(scores.columns):
        raise ValueError("Experimental score columns are incomplete.")

    if scores.duplicated(["metric", "method"]).any():
        raise ValueError("Duplicate experimental scores.")

    options = {
        label: details[0] for label, details in METRICS.items()
    }
    options["Wind direction"] = "wind_direction_deg"

    selected = st.selectbox("Metric to compare", list(options))
    metric = options[selected]
    rows = scores.loc[scores["metric"].eq(metric)].copy()

    if set(rows["method"]) != {
        "selected_models", "persistence", "previous_day"
    }:
        raise ValueError("Expected model and both baseline scores.")

    count_columns = [
        "scored_hours", "reference_pending_hours",
        "excluded_other_hours", "eligible_elapsed_hours",
    ]
    for column in count_columns:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if rows[column].isna().any() or rows[column].lt(0).any():
            raise ValueError("Invalid evaluation counts.")
        if rows[column].nunique() != 1:
            raise ValueError("Methods were not evaluated on matching counts.")

    count = int(rows["scored_hours"].iloc[0])
    pending = int(rows["reference_pending_hours"].iloc[0])
    excluded = int(rows["excluded_other_hours"].iloc[0])

    cards = st.columns(3)
    cards[0].metric("Hours scored", count)
    cards[1].metric("Reference pending", pending)
    cards[2].metric("Other exclusions", excluded)

    if not count:
        st.info("Reference pending or no comparable hours. No score is available.")
        return

    measure = (
        "brier_score"
        if metric == "rain_probability"
        else st.radio("Error measure", ["rmse", "mae"], horizontal=True)
    )
    rows[measure] = pd.to_numeric(rows[measure], errors="raise")
    if not np.isfinite(rows[measure].to_numpy(float)).all():
        raise ValueError("Missing or invalid scores for evaluated hours.")

    rows["Method"] = rows["method"].map(MODEL_NAMES)
    label = "Brier score" if measure == "brier_score" else measure.upper()
    if metric == "wind_direction_deg":
        label += " (angular degrees)"

    st.write("**Lower is better.** All three methods use the same eligible hours.")
    figure = px.bar(
        rows,
        x="Method",
        y=measure,
        color_discrete_sequence=["#2563EB"],
    )
    style_chart(figure, label)
    figure.update_layout(hovermode="closest")
    st.plotly_chart(figure, width="stretch")

    columns = (
        ["Method", "scored_hours", "brier_score"]
        if metric == "rain_probability"
        else ["Method", "scored_hours", "mae", "rmse"]
    )
    st.dataframe(rows[columns].round(4), hide_index=True, width="stretch")
    st.caption(
        "Partial results from one forecast run do not establish reliability. "
        "The evaluator excludes intervals starting before forecast generation."
    )


def historical_performance():
    st.subheader("Historical model performance")
    st.write(
        "October 2026 held-out testing: 720 prediction times, "
        "with forecast horizons from 1 to 24 hours."
    )
    st.caption(
        "Models were selected on earlier validation data. "
        "These results do not validate the recent GFS-input experiment."
    )

    summary = pd.read_csv(TEST_DIR / "regression_summary.csv")
    scores = pd.read_csv(TEST_DIR / "regression_by_horizon.csv")
    options = {
        label: details
        for label, details in METRICS.items()
        if details[0] in set(summary["metric"])
    }
    selected = st.selectbox("Metric to compare", list(options))
    metric, unit = options[selected]
    measure = st.radio("Error measure", ["RMSE", "MAE"], horizontal=True)

    rows = summary.loc[summary["metric"].eq(metric)].copy()
    rows["Model"] = rows["model"].map(
        lambda value: MODEL_NAMES.get(value, value)
    )
    column = f"mean_{measure.lower()}"

    st.write("**Lower error is better.**")
    figure = px.bar(
        rows, x="Model", y=column,
        color_discrete_sequence=["#2563EB"],
    )
    style_chart(figure, f"Mean {measure} across horizons — {unit}")
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        rows[["Model", "mean_rmse", "mean_mae"]].round(4),
        hide_index=True,
        width="stretch",
    )

    with st.expander("Errors by forecast horizon"):
        detail = scores.loc[scores["metric"].eq(metric)].copy()
        detail["Model"] = detail["model"].map(
            lambda value: MODEL_NAMES.get(value, value)
        )
        figure = px.line(
            detail.sort_values("horizon_hours"),
            x="horizon_hours",
            y=measure.lower(),
            color="Model",
        )
        style_chart(figure, measure)
        st.plotly_chart(figure, width="stretch")

    st.info(
        "Test performance was mixed. Selected models did not consistently "
        "beat simple baselines. These test results were not used for retuning."
    )

    with st.expander("Rain probability and wind direction"):
        for filename, title in (
            ("rain_summary.csv", "Rain probability"),
            ("wind_direction_summary.csv", "Wind direction"),
        ):
            st.markdown(f"**{title}**")
            st.dataframe(
                pd.read_csv(TEST_DIR / filename),
                hide_index=True,
                width="stretch",
            )
        st.caption(
            "Lower Brier scores and angular errors are better. "
            "Rain-probability calibration is not established."
        )


def historical_analysis():
    st.subheader("Historical patterns")
    st.write(
        "Training data: November 2025 to July 2026. "
        "ERA5 weather and CAMS modelled air quality."
    )
    choices = {
        "Seasonal trends": (
            "02_daily_trends.png",
            "Daily summaries show changes across the training period.",
        ),
        "Daily patterns": (
            "03_hourly_patterns.png",
            (
                "Karachi-hour averages include abrupt temperature and humidity "
                "patterns present in the source data."
            ),
        ),
        "Value distributions": (
            "01_distributions.png",
            "Rain was rare, limiting the evidence available for rainfall modelling.",
        ),
    }
    selected = st.selectbox("Choose an analysis", list(choices))
    filename, explanation = choices[selected]
    st.write(explanation)
    path = EDA_DIR / filename
    if path.is_file():
        st.image(str(path), width="stretch")
    else:
        st.warning(f"Saved chart not found: {filename}")
    st.caption("Historical training summaries, not current observations.")


def show_period(frame, origin, generated, experimental):
    first = frame["valid_time_utc"].iloc[0]
    last = frame["valid_time_utc"].iloc[-1]
    st.write(f"**Prediction period:** {format_time(first)} to {format_time(last)}")
    st.caption(
        f"Input cutoff: {format_time(origin)}. "
        f"Generated: {format_time(generated)}."
    )
    if experimental:
        now = pd.Timestamp.now(tz="UTC")
        generated_time = pd.Timestamp(generated)
        age_hours = (now - generated_time).total_seconds() / 3600
        input_age = (now - pd.Timestamp(origin)).total_seconds() / 3600
        st.caption(
            f"Forecast age: {age_hours:.1f} hours. "
            f"Input cutoff age: {input_age:.1f} hours."
        )
        if age_hours < 0:
            raise ValueError("Forecast generation time is in the future.")
        if age_hours > 3:
            st.warning(
                "Stale results: this forecast was generated more than 3 hours ago. "
                "A newer forecast has not been published here. "
                "This freshness threshold is a display rule, not an accuracy test."
            )
        if now > last:
            st.warning("This saved prediction period has ended.")
        else:
            st.caption(
                "This is a saved experimental forecast. Refresh reads files; "
                "it does not collect weather or generate a new forecast."
            )


def main():
    st.set_page_config(page_title="AgentWeather | Karachi", layout="wide")
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stHeader"] {
            background-color: #F5F7FB; color: #172B4D;
        }
        [data-testid="stSidebar"] { background-color: #EAF0F8; }
        [data-testid="stMetric"] {
            background-color: white; border: 1px solid #DCE5F0;
            border-radius: 12px; padding: 18px; margin-bottom: 12px;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: #172B4D;
        }
        h1, h2, h3 { color: #122746 !important; }
        .block-container { padding-top: 2.5rem; max-width: 1250px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("AgentWeather")
    mode = st.sidebar.radio(
        "Data view", ["Historical demo", "Experimental recent inputs"]
    )
    page = st.sidebar.radio(
        "Explore",
        [
            "Weather Overview",
            "Alerts and Report",
            "Model Performance",
            "Historical Analysis",
        ],
    )
    st.sidebar.button("Refresh displayed results")
    st.sidebar.caption(
        "Displays published forecast files. GitHub runs the models separately. "
        "Refreshing this page does not generate a forecast."
    )

    st.title("AgentWeather")
    st.caption("Weather and air-quality insights for Karachi")

    try:
        if page == "Historical Analysis":
            historical_analysis()
            return

        if mode == "Historical demo":
            st.info("Historical demonstration — not today's weather or warnings.")

            if page == "Model Performance":
                historical_performance()
                return

            log, frame, alerts, monitoring, report = load_historical()
            show_period(
                frame, log["origin_utc"], log["started_at_utc"], False
            )
            if page == "Weather Overview":
                overview(frame)
            else:
                historical_report(alerts, report)

            with st.expander("System Status"):
                st.write("Workflow:", log["status"])
                st.write("Input quality passed:", monitoring["quality_passed"])
                st.write("Monitoring:", monitoring["status"])
                st.caption("Input distribution screening is not an accuracy guarantee.")
            return

        st.warning(
            "Experimental — accuracy with recent GFS inputs is not established. "
            "These predictions are not observed conditions or current warnings."
        )
        runs = list_experimental_runs()
        if not runs:
            st.info(
                "No successful experimental pipeline run was found. "
                "Run python run_experimental_pipeline.py in your terminal."
            )
            return

        run_dir = st.sidebar.selectbox(
            "Saved experimental run",
            runs,
            format_func=lambda path: path.name,
        )
        log, frame = load_experimental(run_dir)
        if run_dir.parent == CLOUD_DIR:
            st.caption("Source: published cloud workflow output.")
        else:
            st.info("Bundled demonstration run; this is not the latest cloud feed.")
        show_period(
            frame, log["input_cutoff_utc"], log["generated_at_utc"], True
        )

        if page == "Weather Overview":
            overview(frame)
        elif page == "Alerts and Report":
            experimental_report(frame)
        else:
            experimental_performance(run_dir)

        with st.expander("System Status"):
            st.write("Workflow:", log["status"])
            st.write("Forecast rows:", len(frame))
            st.write("Recent weather inputs:", "GFS Global")
            st.write("Older weather history:", "ERA5")
            st.write("Air-quality inputs:", "CAMS Global")
            st.write(
                "Weather-source transition:",
                format_time(log["source_transition_utc"]),
            )
            st.caption(
                "Models remain fixed. Experimental alerts and notifications "
                "are disabled. No automatic retraining is implemented."
            )

    except ERRORS as error:
        st.error(f"Saved results could not be displayed: {error}")


if __name__ == "__main__":
    main()