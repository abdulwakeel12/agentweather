"""Run historical monitoring, forecasts, alerts and reporting."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from alert_agent import evaluate_alerts
from build_features import make_features
from drift_agent import assess_drift
from forecast import AgentWeatherForecaster, read_history
from report_agent import generate_report

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = (
    BASE_DIR / "data" / "inference" / "history_through_2026-11-15.csv"
)
MODE = "retrospective_historical_demo"


class AgentState(TypedDict, total=False):
    run_id: str
    started_at_utc: str
    finished_at_utc: str
    status: str
    mode: str
    history_path: str
    origin_utc: str
    input_age_hours: float
    model_versions: dict
    monitoring_path: str
    monitoring_status: str
    monitoring_quality_passed: bool
    flagged_metrics: list
    forecast_path: str
    forecast_rows: int
    alert_path: str
    alert_count: int
    alerts: list
    report_path: str
    report_method: str
    log_path: str
    error: str
    events: list


def event(node, status, message):
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "node": node,
        "status": status,
        "message": message,
    }


def failed(state, node, error):
    message = f"{type(error).__name__}: {error}"
    print(f"\n{node} failed: {message}", flush=True)

    return {
        "status": "failed",
        "error": message,
        "events": [
            *state.get("events", []),
            event(node, "failed", message),
        ],
    }


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)

    temporary.replace(path)


def save_json(path, content):
    text = json.dumps(
        content,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    save_text(path, text + "\n")


def route_after_step(state):
    return "stop" if state["status"] == "failed" else "continue"


def validate_forecast(frame, origin):
    if len(frame) != 24:
        raise ValueError("Expected exactly 24 forecast rows.")

    if frame["horizon_hours"].tolist() != list(range(1, 25)):
        raise ValueError("Forecast horizons must be ordered from 1 to 24.")

    prediction_times = pd.to_datetime(
        frame["prediction_time_utc"],
        utc=True,
        errors="raise",
    )
    if not prediction_times.eq(origin).all():
        raise ValueError("Forecast prediction origins do not match.")

    valid_times = pd.DatetimeIndex(
        pd.to_datetime(
            frame["valid_time_utc"],
            utc=True,
            errors="raise",
        )
    )
    expected_times = pd.date_range(
        start=origin + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
    )
    if not valid_times.equals(expected_times):
        raise ValueError("Forecast target timestamps are incorrect.")

    metrics = [
        "us_aqi",
        "temperature_c",
        "precipitation_mm",
        "humidity_pct",
        "uv_index",
        "wind_speed_ms",
        "rain_probability",
    ]
    if not np.isfinite(frame[metrics].to_numpy(dtype=float)).all():
        raise ValueError("Forecast contains missing or infinite values.")

    for metric in (
        "us_aqi",
        "precipitation_mm",
        "uv_index",
        "wind_speed_ms",
    ):
        if frame[metric].lt(0).any():
            raise ValueError(f"Negative forecast values: {metric}")

    if not frame["humidity_pct"].between(0, 100).all():
        raise ValueError("Humidity is outside 0–100.")

    if not frame["rain_probability"].between(0, 1).all():
        raise ValueError("Rain probability is outside 0–1.")

    flags = frame["wind_direction_vector_valid"]
    if flags.isna().any() or not flags.isin([True, False]).all():
        raise ValueError("Invalid wind-direction validity flags.")

    valid = flags.to_numpy(dtype=bool)
    directions = frame["wind_direction_deg"].to_numpy(dtype=float)

    if not np.isfinite(directions[valid]).all():
        raise ValueError("Valid wind directions must be finite.")

    if not np.isnan(directions[~valid]).all():
        raise ValueError("Undefined wind directions must be missing.")

    if ((directions[valid] < 0) | (directions[valid] >= 360)).any():
        raise ValueError("Wind directions must be in [0, 360).")


def build_workflow():
    # Loaded models and dataframes are not serialized into the JSON log.
    context = {}

    def monitor_inputs(state):
        print("\nStep 1/6: Monitor historical inputs", flush=True)

        try:
            if state["mode"] != MODE:
                raise ValueError("Only historical demonstration mode is supported.")

            history_path = Path(state["history_path"])

            # Detect changes to the history file between monitoring and loading.
            before = history_path.stat()
            report = assess_drift(history_path)
            after = history_path.stat()

            signature = (after.st_size, after.st_mtime_ns)
            if signature != (before.st_size, before.st_mtime_ns):
                raise ValueError("History changed during monitoring.")

            if report["mode"] != state["mode"]:
                raise ValueError("Monitoring mode does not match.")

            context["history_signature"] = signature

            report["agent_run_id"] = state["run_id"]

            output = (
                BASE_DIR
                / "data"
                / "monitoring"
                / f"agent_monitoring_{state['run_id']}.json"
            )
            save_json(output, report)

            quality_passed = bool(report["quality_passed"])
            flagged = list(report["flagged_metrics"])

            updates = {
                "monitoring_path": str(output),
                "monitoring_status": report["status"],
                "monitoring_quality_passed": quality_passed,
                "flagged_metrics": flagged,
                "origin_utc": report["window_end_utc"],
                "input_age_hours": float(report["input_age_hours"]),
            }

            print(f"Input quality passed: {quality_passed}")
            print(f"Monitoring status: {report['status']}")
            print(f"Flagged metrics: {', '.join(flagged) or 'None'}")
            print(f"Monitoring report: {output}")

            if not quality_passed:
                message = (
                    "Monitoring found missing hours or invalid metric values. "
                    "Forecasting was stopped."
                )
                return {
                    **updates,
                    "status": "failed",
                    "error": message,
                    "events": [
                        *state.get("events", []),
                        event("monitor_inputs", "failed", message),
                    ],
                }

            if flagged:
                message = (
                    "Input quality passed. Distribution flags require review; "
                    "historical forecasting continues without retraining."
                )
                event_status = "warning"
            else:
                message = (
                    "Input quality passed; no initial distribution "
                    "screening thresholds were crossed."
                )
                event_status = "success"

            return {
                **updates,
                "status": "monitoring_completed",
                "events": [
                    *state.get("events", []),
                    event("monitor_inputs", event_status, message),
                ],
            }

        except Exception as error:  # noqa: BLE001
            # Catch node failures so the workflow can record them.
            return failed(state, "monitor_inputs", error)

    def prepare_inputs(state):
        print("\nStep 2/6: Prepare historical features and models", flush=True)

        try:
            history_path = Path(state["history_path"])
            before = history_path.stat()

            if (
                before.st_size,
                before.st_mtime_ns,
            ) != context["history_signature"]:
                raise ValueError("History changed after monitoring.")

            history = read_history(history_path)
            after = history_path.stat()

            if (
                after.st_size,
                after.st_mtime_ns,
            ) != context["history_signature"]:
                raise ValueError("History changed while loading.")

            features = make_features(history).tail(1)

            if len(features) != 1:
                raise ValueError("Expected one inference feature row.")

            origin = features.index[0]
            if origin != pd.Timestamp(state["origin_utc"]):
                raise ValueError("Feature origin differs from monitored history.")

            age_hours = (
                pd.Timestamp.now(tz="UTC") - origin
            ).total_seconds() / 3600

            if age_hours < 0:
                raise ValueError("Prediction origin is in the future.")

            forecaster = AgentWeatherForecaster(project_dir=BASE_DIR)

            context["history"] = history.loc[:origin]
            context["features"] = features
            context["forecaster"] = forecaster

            print(f"Prediction origin: {origin}")
            print(f"Input age: {age_hours:.1f} hours")
            print("Mode: historical demonstration")

            return {
                "status": "prepared",
                "origin_utc": origin.isoformat(),
                "input_age_hours": age_hours,
                "model_versions": forecaster.plan["selected_models"],
                "events": [
                    *state.get("events", []),
                    event(
                        "prepare_inputs",
                        "success",
                        "Historical features and saved models prepared.",
                    ),
                ],
            }

        except Exception as error:  # noqa: BLE001
            return failed(state, "prepare_inputs", error)

    def generate_forecast(state):
        print("\nStep 3/6: Generate forecasts", flush=True)

        try:
            origin = pd.Timestamp(state["origin_utc"])
            frame = context["forecaster"].forecast(
                origin,
                context["features"],
                context["history"],
            )

            validate_forecast(frame, origin)

            frame["generated_at_utc"] = datetime.now(UTC).isoformat()
            frame["forecast_mode"] = state["mode"]
            frame["agent_run_id"] = state["run_id"]

            output = (
                BASE_DIR
                / "data"
                / "forecasts"
                / f"agent_forecast_{state['run_id']}.csv"
            )
            save_text(output, frame.to_csv(index=False))

            return {
                "status": "forecast_completed",
                "forecast_path": str(output),
                "forecast_rows": len(frame),
                "events": [
                    *state.get("events", []),
                    event(
                        "generate_forecast",
                        "success",
                        "Generated and saved 24 hourly forecasts.",
                    ),
                ],
            }

        except Exception as error:  # noqa: BLE001
            return failed(state, "generate_forecast", error)

    def evaluate_forecast_alerts(state):
        print("\nStep 4/6: Evaluate historical alerts", flush=True)

        try:
            frame = pd.read_csv(state["forecast_path"])

            if not frame["agent_run_id"].eq(state["run_id"]).all():
                raise ValueError("Forecast belongs to another workflow run.")

            report = evaluate_alerts(frame)

            if (
                pd.Timestamp(report["prediction_time_utc"])
                != pd.Timestamp(state["origin_utc"])
            ):
                raise ValueError("Alert prediction origin does not match.")

            report["agent_run_id"] = state["run_id"]
            report["forecast_file"] = state["forecast_path"]
            report["input_age_hours_at_preparation"] = state["input_age_hours"]

            output = (
                BASE_DIR
                / "data"
                / "alerts"
                / f"agent_alerts_{state['run_id']}.json"
            )
            save_json(output, report)

            alerts = report["alerts"]
            print(f"Historical alert windows: {len(alerts)}")

            for alert in alerts:
                print(
                    f"  {alert['metric']}: {alert['category_at_peak']} "
                    f"({alert['flagged_hours']} flagged hours)"
                )

            return {
                "status": "alerts_completed",
                "alert_path": str(output),
                "alert_count": len(alerts),
                "alerts": alerts,
                "events": [
                    *state.get("events", []),
                    event(
                        "evaluate_alerts",
                        "success",
                        f"Saved {len(alerts)} historical alert windows.",
                    ),
                ],
            }

        except Exception as error:  # noqa: BLE001
            return failed(state, "evaluate_alerts", error)

    def write_weather_report(state):
        print("\nStep 5/6: Write historical weather report", flush=True)

        try:
            report = generate_report(
                Path(state["forecast_path"]),
                Path(state["alert_path"]),
            )

            if report["agent_run_id"] != state["run_id"]:
                raise ValueError("Report belongs to another workflow run.")

            if report["mode"] != state["mode"]:
                raise ValueError("Report mode does not match.")

            if (
                pd.Timestamp(report["prediction_time_utc"])
                != pd.Timestamp(state["origin_utc"])
            ):
                raise ValueError("Report prediction origin does not match.")

            flagged_text = ", ".join(state["flagged_metrics"]) or "None"
            monitoring_text = "\n".join(
                [
                    "",
                    "## Input monitoring",
                    "",
                    f"Monitoring status: `{state['monitoring_status']}`",
                    f"Flagged metrics: {flagged_text}",
                    (
                        "Input age when prepared: "
                        f"{state['input_age_hours']:.1f} hours."
                    ),
                    "",
                    (
                        "Monitoring uses initial distribution-screening "
                        "rules. Seasonal differences can produce flags. "
                        "No flags do not establish unchanged model accuracy."
                    ),
                    (
                        "Forecast-error drift was not assessed. "
                        "No retraining was triggered."
                    ),
                    "",
                ]
            )

            output = (
                BASE_DIR
                / "data"
                / "reports"
                / f"agent_report_{state['run_id']}.md"
            )
            save_text(output, report["text"] + monitoring_text)

            print("Report method: factual template")
            print(f"Report saved: {output}")

            return {
                "status": "report_completed",
                "report_path": str(output),
                "report_method": report["report_method"],
                "events": [
                    *state.get("events", []),
                    event(
                        "write_report",
                        "success",
                        "Saved weather report with input-monitoring summary.",
                    ),
                ],
            }

        except Exception as error:  # noqa: BLE001
            return failed(state, "write_report", error)

    def save_execution_log(state):
        print("\nFinal step: Save execution log (6/6 on success)", flush=True)

        status = (
            "success"
            if state["status"] == "report_completed"
            else "failed"
        )
        output = (
            BASE_DIR
            / "data"
            / "agent_logs"
            / f"{state['run_id']}.json"
        )

        updates = {
            "status": status,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "log_path": str(output),
            "events": [
                *state.get("events", []),
                event(
                    "save_log",
                    status,
                    "Workflow finished; execution details recorded.",
                ),
            ],
        }

        save_json(output, {**state, **updates})
        print(f"Execution log: {output}")

        return updates

    graph = StateGraph(AgentState)

    graph.add_node("monitor_inputs", monitor_inputs)
    graph.add_node("prepare_inputs", prepare_inputs)
    graph.add_node("generate_forecast", generate_forecast)
    graph.add_node("evaluate_alerts", evaluate_forecast_alerts)
    graph.add_node("write_report", write_weather_report)
    graph.add_node("save_log", save_execution_log)

    graph.add_edge(START, "monitor_inputs")

    transitions = [
        ("monitor_inputs", "prepare_inputs"),
        ("prepare_inputs", "generate_forecast"),
        ("generate_forecast", "evaluate_alerts"),
        ("evaluate_alerts", "write_report"),
    ]

    for current_node, next_node in transitions:
        graph.add_conditional_edges(
            current_node,
            route_after_step,
            {
                "continue": next_node,
                "stop": "save_log",
            },
        )

    graph.add_edge("write_report", "save_log")
    graph.add_edge("save_log", END)

    return graph.compile()


def main():
    started_at = datetime.now(UTC)
    run_id = (
        started_at.strftime("%Y%m%dT%H%M%S%f")
        + "_"
        + uuid4().hex[:8]
    )

    initial_state = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "status": "started",
        "mode": MODE,
        "history_path": str(HISTORY_PATH),
        "events": [],
    }

    print("AgentWeather — Historical Four-Component Workflow")

    try:
        workflow = build_workflow()
        result = workflow.invoke(initial_state)

    except Exception as error:  # noqa: BLE001
        # Includes graph construction and execution-log write failures.
        print(
            "\nWorkflow could not finish: "
            f"{type(error).__name__}: {error}"
        )
        print(f"Run ID: {run_id}")
        print("Any outputs already written have been preserved.")
        return 1

    if result["status"] != "success":
        print(f"\nWorkflow failed: {result.get('error', 'Unknown error')}")
        print(f"Execution log: {result.get('log_path', 'Unavailable')}")
        return 1

    print("\nHistorical workflow completed successfully.")
    print(f"Monitoring status: {result['monitoring_status']}")
    print(f"Forecast rows: {result['forecast_rows']}")
    print(f"Historical alert windows: {result['alert_count']}")
    print(f"Monitoring JSON: {result['monitoring_path']}")
    print(f"Forecast CSV: {result['forecast_path']}")
    print(f"Alert JSON: {result['alert_path']}")
    print(f"Weather report: {result['report_path']}")
    print(f"Execution log: {result['log_path']}")
    print("Historical demonstration only—not current warnings.")
    print("Report uses a fixed template; no LLM calls.")
    print("Rain alerts disabled. No notifications sent.")
    print("Forecast-error drift and automatic retraining are not implemented.")
    print("No models changed or uploaded to Hopsworks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())