"""AgentWeather dashboard for saved historical results."""

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from report_agent import read_forecast

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = (
    DATA_DIR / "evaluation" / "final_test_20261121T112620710620"
)
EDA_DIR = DATA_DIR / "eda" / "training"
MODE = "retrospective_historical_demo"

METRICS = {
    "Temperature": ("temperature_c", "Temperature (°C)"),
    "Humidity": ("humidity_pct", "Humidity (%)"),
    "Air quality": ("us_aqi", "Modelled US AQI"),
    "UV Index": ("uv_index", "UV Index"),
    "Wind speed": ("wind_speed_ms", "Wind speed (m/s)"),
    "Precipitation": ("precipitation_mm", "Hourly precipitation (mm)"),
    "Rain probability": ("rain_probability_pct", "Hourly rain probability (%)"),
}

MODEL_NAMES = {
    "random_forest": "Random Forest",
    "lstm": "LSTM",
    "sarimax": "SARIMAX",
    "ridge": "Ridge",
    "previous_day": "Previous-day baseline",
    "zero_precipitation": "Zero-rainfall baseline",
    "training_mean_precipitation": "Training-average baseline",
    "always_dry": "Always-dry baseline",
    "training_frequency": "Training-frequency baseline",
    "logistic": "Logistic regression",
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path.name}")
    return value


def local_time(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Timestamp must contain a timezone.")
    return timestamp.tz_convert("Asia/Karachi")


def format_time(value):
    return local_time(value).strftime("%d %b %Y, %H:%M PKT")


def load_latest_run():
    logs = sorted(
        (DATA_DIR / "agent_logs").glob("*.json"),
        reverse=True,
    )
    skipped = 0

    for path in logs:
        try:
            log = read_json(path)
        except (OSError, ValueError, TypeError):
            skipped += 1
            continue

        if (
            log.get("status") != "success"
            or log.get("mode") != MODE
            or not log.get("monitoring_path")
            or not log.get("report_path")
        ):
            skipped += 1
            continue

        run_id = log["run_id"]
        if not isinstance(run_id, str) or not re.fullmatch(
            r"\d{8}T\d{12}_[a-f0-9]{8}", run_id
        ):
            raise ValueError("Invalid workflow identifier.")

        if path.stem != run_id:
            raise ValueError("Execution log identifier does not match.")

        # Use one run's outputs together, never separate latest files.
        forecast_path = (
            DATA_DIR / "forecasts" / f"agent_forecast_{run_id}.csv"
        )
        alert_path = DATA_DIR / "alerts" / f"agent_alerts_{run_id}.json"
        monitor_path = (
            DATA_DIR / "monitoring" / f"agent_monitoring_{run_id}.json"
        )
        report_path = DATA_DIR / "reports" / f"agent_report_{run_id}.md"

        rows, origin, forecast_run = read_forecast(forecast_path)
        if forecast_run != run_id:
            raise ValueError("Forecast and execution log do not match.")
        if pd.Timestamp(log["origin_utc"]) != pd.Timestamp(origin):
            raise ValueError("Forecast origins do not match.")

        alerts = read_json(alert_path)
        monitoring = read_json(monitor_path)

        for document in (alerts, monitoring):
            if document.get("agent_run_id") != run_id:
                raise ValueError("Saved outputs belong to different runs.")
            if document.get("mode") != MODE:
                raise ValueError("Saved output is not marked historical.")

        if pd.Timestamp(alerts["prediction_time_utc"]) != pd.Timestamp(origin):
            raise ValueError("Alert origin does not match.")
        if pd.Timestamp(monitoring["window_end_utc"]) != pd.Timestamp(origin):
            raise ValueError("Monitoring origin does not match.")

        report = report_path.read_text(encoding="utf-8")
        if run_id not in report:
            raise ValueError("Weather report identifier does not match.")

        frame = pd.DataFrame(rows)
        frame["Time (PKT)"] = (
            pd.to_datetime(frame["valid_time_utc"], utc=True)
            .dt.tz_convert("Asia/Karachi")
            .dt.tz_localize(None)
        )
        frame["rain_probability_pct"] = frame["rain_probability"] * 100

        return log, frame, alerts, monitoring, report, skipped

    raise FileNotFoundError(
        "No completed four-component workflow was found. "
        "Run python agent_workflow.py first."
    )


def chart_style(figure, y_label):
    figure.update_layout(
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#172B4D", "size": 13},
        margin={"l": 20, "r": 20, "t": 25, "b": 25},
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title_text="",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#E8EEF5", zeroline=False)
    return figure


def overview(frame):
    st.subheader("Karachi forecast overview")
    st.caption("Each card summarizes the full 24-hour prediction period.")

    first = st.columns(3)
    first[0].metric(
        "Temperature range",
        f"{frame['temperature_c'].min():.1f}–"
        f"{frame['temperature_c'].max():.1f} °C",
    )
    first[1].metric(
        "Highest modelled AQI",
        f"{frame['us_aqi'].max():.1f}",
    )
    first[2].metric(
        "Total predicted precipitation",
        f"{frame['precipitation_mm'].sum():.2f} mm",
    )

    second = st.columns(3)
    second[0].metric(
        "Humidity range",
        f"{frame['humidity_pct'].min():.0f}–"
        f"{frame['humidity_pct'].max():.0f}%",
    )
    second[1].metric("Highest UV Index", f"{frame['uv_index'].max():.2f}")
    second[2].metric(
        "Highest wind speed",
        f"{frame['wind_speed_ms'].max():.1f} m/s",
    )

    st.divider()
    st.subheader("Explore the next 24 forecast hours")

    selection = st.selectbox("Weather metric", list(METRICS))
    column, label = METRICS[selection]

    arguments = {
        "data_frame": frame,
        "x": "Time (PKT)",
        "y": column,
        "labels": {column: label},
        "color_discrete_sequence": ["#2563EB"],
    }

    if column == "precipitation_mm":
        figure = px.bar(**arguments)
    else:
        figure = px.line(**arguments, markers=True)

    chart_style(figure, label)

    if column in ("humidity_pct", "rain_probability_pct"):
        figure.update_yaxes(range=[0, 100])
    elif column != "temperature_c":
        figure.update_yaxes(rangemode="tozero")

    st.plotly_chart(figure, width="stretch")

    if column == "rain_probability_pct":
        st.caption(
            "Probability applies to each individual hour, not the whole day. "
            "Calibration is not established; rain alerts remain disabled."
        )
    elif column == "us_aqi":
        st.caption(
            "Predictions use modelled air-quality data, "
            "not measurements from a Karachi monitoring station."
        )
    elif column == "wind_speed_ms":
        st.caption(
            "Wind direction is not displayed until the policy for "
            "calm or ambiguous direction predictions is defined."
        )
    else:
        st.caption("All times are Karachi time. These are predicted values.")

    with st.expander("View hourly values"):
        table = frame[["Time (PKT)", column]].rename(columns={column: label})
        st.dataframe(table.round(3), hide_index=True, width="stretch")


def alerts_and_report(alerts, report):
    st.subheader("Historical alert summary")
    st.caption("These flags describe the saved forecast, not current warnings.")

    names = {"us_aqi": "Air quality", "uv_index": "UV exposure"}
    windows = alerts["alerts"]

    if not windows:
        st.write("No AQI or UV alert windows were identified.")
        st.caption("This does not establish that all conditions are safe.")

    for alert in windows:
        with st.container(border=True):
            st.markdown(
                f"**{names.get(alert['metric'], alert['metric'])}: "
                f"{alert['category_at_peak']}**"
            )
            st.write(
                f"{format_time(alert['first_target_time_utc'])} "
                f"to {format_time(alert['last_target_time_utc'])}"
            )
            st.caption(
                f"{alert['flagged_hours']} flagged hours. "
                f"Peak predicted value: {alert['peak_predicted_value']:.2f}. "
                "The category shown applies to the window peak."
            )

    st.caption("Rain alerts are disabled. No notifications were sent.")
    st.divider()
    st.subheader("Weather report")
    st.caption("Generated from the saved values using a factual template.")

    # Keep the weather narrative; technical details remain in System Status.
    weather_text = report.split("## Weather outlook", 1)
    if len(weather_text) == 2:
        narrative = weather_text[1].split("## Historical alert evaluation", 1)[0]
        st.markdown(narrative)
    else:
        st.markdown(report)

    st.download_button(
        "Download complete report",
        data=report,
        file_name="AgentWeather_historical_report.md",
        mime="text/markdown",
    )


def load_table(filename, required):
    frame = pd.read_csv(TEST_DIR / filename)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return frame


def model_performance():
    st.subheader("Model performance")
    st.write(
        "October 2026 was held out for final testing. "
        "Models were selected using earlier validation data."
    )
    st.caption(
        "These results cover 720 prediction times and horizons 1–24 hours. "
        "They are separate from the November forecast shown on Overview."
    )

    try:
        summary = load_table(
            "regression_summary.csv",
            ["metric", "model", "mean_rmse", "mean_mae"],
        )
        scores = load_table(
            "regression_by_horizon.csv",
            ["metric", "model", "horizon_hours", "rmse", "mae"],
        )

        options = {
            label: details
            for label, details in METRICS.items()
            if details[0] in set(summary["metric"])
        }
        if not options:
            raise ValueError("No recognized regression metrics were found.")

        selection = st.selectbox("Metric to compare", list(options))
        metric, unit = options[selection]
        measure = st.radio(
            "Error measure",
            ["RMSE", "MAE"],
            horizontal=True,
        )
        score_column = f"mean_{measure.lower()}"

        selected = summary.loc[summary["metric"].eq(metric)].copy()
        selected["Model"] = selected["model"].map(
            lambda name: MODEL_NAMES.get(name, name)
        )

        st.write(
            "**Lower error is better.** RMSE gives more weight to large "
            "mistakes; MAE is the average absolute prediction error."
        )

        figure = px.bar(
            selected,
            x="Model",
            y=score_column,
            color_discrete_sequence=["#2563EB"],
            labels={score_column: f"Mean {measure} across 24 horizons"},
        )
        chart_style(figure, f"Mean {measure} — {unit}")
        figure.update_layout(hovermode="closest")
        st.plotly_chart(figure, width="stretch")

        table = selected[["Model", "mean_rmse", "mean_mae"]].rename(
            columns={
                "mean_rmse": "Mean RMSE",
                "mean_mae": "Mean MAE",
            }
        )
        st.dataframe(table.round(4), hide_index=True, width="stretch")
        st.caption("Errors retain the selected metric's units.")

        with st.expander("How errors change with forecast horizon"):
            detail = scores.loc[scores["metric"].eq(metric)].copy()
            detail["Model"] = detail["model"].map(
                lambda name: MODEL_NAMES.get(name, name)
            )
            figure = px.line(
                detail.sort_values("horizon_hours"),
                x="horizon_hours",
                y=measure.lower(),
                color="Model",
                labels={"horizon_hours": "Hours ahead"},
            )
            chart_style(figure, measure)
            st.plotly_chart(figure, width="stretch")

        st.info(
            "Test performance was mixed. The selected models did not "
            "consistently beat simple baselines. The test results were "
            "not used to retune or replace them."
        )

        with st.expander("Rain probability and wind-direction results"):
            st.write(
                "For rain probabilities, lower Brier scores are better. "
                "For wind direction, lower angular errors are better."
            )
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
                "Rain probabilities are not established as calibrated. "
                "At the 0.5 threshold, the +24-hour classifier detected "
                "none of the eight rainy target hours in the test period."
            )

    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Could not display saved evaluation results: {error}")


def historical_analysis():
    st.subheader("Historical patterns")
    st.write(
        "Explore the training period: November 2025 to July 2026. "
        "These charts use historical ERA5 weather and CAMS modelled air quality."
    )

    choices = {
        "Seasonal trends": (
            "02_daily_trends.png",
            (
                "Daily summaries show how conditions changed across the "
                "training period. Weather and air quality vary by season."
            ),
        ),
        "Daily patterns": (
            "03_hourly_patterns.png",
            (
                "These averages group observations by Karachi local hour. "
                "Temperature and humidity show abrupt patterns in the source "
                "data; reproducing them does not establish independent accuracy."
            ),
        ),
        "Value distributions": (
            "01_distributions.png",
            (
                "These charts show how frequently different values occurred. "
                "Rain was rare, which makes rainfall modelling difficult."
            ),
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

    st.caption(
        "These are training-data summaries, not live observations. "
        "Validation and test periods were excluded from this EDA."
    )


def main():
    st.set_page_config(
        page_title="AgentWeather | Karachi",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F5F7FB;
            color: #172B4D;
        }
        [data-testid="stHeader"] {
            background-color: #F5F7FB;
        }
        [data-testid="stSidebar"] {
            background-color: #EAF0F8;
        }
        [data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #DCE5F0;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 12px;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #172B4D;
        }
        h1, h2, h3 {
            color: #122746 !important;
        }
        .block-container {
            padding-top: 2.5rem;
            max-width: 1250px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("AgentWeather")
    st.sidebar.caption("Karachi weather intelligence")

    page = st.sidebar.radio(
        "Explore",
        [
            "Weather Overview",
            "Alerts and Report",
            "Model Performance",
            "Historical Analysis",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.button("Refresh results")
    st.sidebar.caption("Historical demonstration")

    st.title("AgentWeather")
    st.caption("Weather and air-quality insights for Karachi")
    st.info(
        "Historical demonstration — this dashboard does not show "
        "today's weather or current warnings."
    )

    if page == "Model Performance":
        model_performance()
        return

    if page == "Historical Analysis":
        historical_analysis()
        return

    try:
        log, frame, alerts, monitoring, report, skipped = load_latest_run()
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Saved forecast could not be loaded: {error}")
        return

    first = frame["Time (PKT)"].iloc[0].strftime("%d %b %Y, %H:%M")
    last = frame["Time (PKT)"].iloc[-1].strftime("%d %b %Y, %H:%M")

    st.write(f"**Forecast period:** {first} to {last} PKT")
    st.caption(
        f"Historical input cutoff: {format_time(log['origin_utc'])}. "
        f"Generated retrospectively: {format_time(log['started_at_utc'])}."
    )

    if skipped:
        st.caption(
            "Showing the latest completed four-component run. "
            "Newer incomplete, incompatible, or unreadable logs were skipped."
        )

    if page == "Weather Overview":
        overview(frame)
    else:
        alerts_and_report(alerts, report)

    st.divider()

    with st.expander("System Status"):
        st.write("Workflow:", log["status"])
        st.write("Input quality passed:", monitoring["quality_passed"])
        st.write("Monitoring:", monitoring["status"])
        st.write(
            "Flagged metrics:",
            ", ".join(monitoring["flagged_metrics"]) or "None",
        )
        st.write(
            "Input age when prepared:",
            f"{log['input_age_hours']:.1f} hours",
        )
        st.caption(
            "Distribution screening is not an accuracy guarantee. "
            "Forecast-error drift and automatic retraining are not implemented."
        )


if __name__ == "__main__":
    main()