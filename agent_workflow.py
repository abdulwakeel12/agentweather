import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from build_features import make_features
from forecast import AgentWeatherForecaster, read_history

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = (
    BASE_DIR / "data/inference/history_through_2026-11-15.csv"
)


class AgentState(TypedDict, total=False):
    run_id: str
    started_at_utc: str
    status: str
    mode: str
    history_path: str
    origin_utc: str
    input_age_hours: float
    model_versions: dict
    forecast_path: str
    forecast_rows: int
    error: str
    events: list[dict]


def event(node, status, message):
    return {
        "time_utc": datetime.now(UTC).isoformat(),
        "node": node,
        "status": status,
        "message": message,
    }


def failed(state, node, error):
    message = f"{type(error).__name__}: {error}"
    print(f"{node} failed: {message}", flush=True)
    return {
        "status": "failed",
        "error": message,
        "events": state.get("events", []) + [
            event(node, "failed", message)
        ],
    }


def build_workflow():
    # Keep large dataframes outside the serializable graph state.
    # This context belongs to this single workflow execution.
    context = {}

    def prepare_inputs(state):
        print("\nStep 1/3: Prepare historical inputs", flush=True)
        try:
            history = read_history(Path(state["history_path"]))
            features = make_features(history).tail(1)
            origin = features.index[0]

            age = (
                datetime.now(UTC) - origin.to_pydatetime()
            ).total_seconds() / 3600
            if age < 0:
                raise ValueError("The historical origin is in the future.")

            forecaster = AgentWeatherForecaster(project_dir=BASE_DIR)
            context.update({
                "history": history.loc[:origin],
                "features": features,
                "forecaster": forecaster,
            })

            print(f"Prediction origin: {origin}")
            print(f"Input age: {age:.1f} hours")
            print("Mode: historical demonstration, not live forecasting")

            return {
                "status": "prepared",
                "origin_utc": origin.isoformat(),
                "input_age_hours": age,
                "model_versions": forecaster.plan["selected_models"],
                "events": state["events"] + [
                    event(
                        "prepare_inputs",
                        "success",
                        "Prepared historical inputs; live mode is not enabled.",
                    )
                ],
            }
        except Exception as error:
            return failed(state, "prepare_inputs", error)

    def generate_forecast(state):
        print("\nStep 2/3: Generate forecasts", flush=True)
        try:
            origin = pd.Timestamp(state["origin_utc"])
            result = context["forecaster"].forecast(
                origin,
                context["features"],
                context["history"],
            )

            if result["horizon_hours"].tolist() != list(range(1, 25)):
                raise ValueError("Expected exactly 24 ordered forecast horizons.")

            expected_times = pd.date_range(
                origin + pd.Timedelta(hours=1), periods=24, freq="h"
            )
            if not pd.DatetimeIndex(result["valid_time_utc"]).equals(
                expected_times
            ):
                raise ValueError("Forecast target timestamps differ.")

            required = [
                "us_aqi", "temperature_c", "precipitation_mm",
                "humidity_pct", "uv_index", "wind_speed_ms",
                "rain_probability",
            ]
            if not np.isfinite(result[required].to_numpy(float)).all():
                raise ValueError("Forecast contains invalid numeric values.")

            if not result["rain_probability"].between(0, 1).all():
                raise ValueError("Rain probabilities must be between 0 and 1.")

            # Undefined wind directions are allowed only for invalid vectors.
            valid = result["wind_direction_vector_valid"].to_numpy(bool)
            angles = result["wind_direction_deg"].to_numpy(float)
            if (
                not np.isfinite(angles[valid]).all()
                or not np.isnan(angles[~valid]).all()
            ):
                raise ValueError("Wind direction validity flags disagree.")

            result["generated_at_utc"] = datetime.now(UTC).isoformat()
            result["forecast_mode"] = state["mode"]
            result["agent_run_id"] = state["run_id"]

            folder = BASE_DIR / "data/forecasts"
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / f"agent_forecast_{state['run_id']}.csv"
            temporary = output.with_suffix(".tmp")
            result.to_csv(temporary, index=False)
            temporary.replace(output)

            return {
                "status": "forecast_completed",
                "forecast_path": str(output),
                "forecast_rows": len(result),
                "events": state["events"] + [
                    event(
                        "generate_forecast",
                        "success",
                        "Saved 24 historical forecast rows. No alerts issued.",
                    )
                ],
            }
        except Exception as error:
            return failed(state, "generate_forecast", error)

    def save_log(state):
        print("\nStep 3/3: Save execution log", flush=True)
        status = (
            "success" if state["status"] == "forecast_completed"
            else "failed"
        )
        events = state["events"] + [
            event("save_log", status, "Workflow execution finished.")
        ]
        log = {
            **state,
            "status": status,
            "events": events,
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }

        folder = BASE_DIR / "data/agent_logs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{state['run_id']}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(log, file, indent=2, allow_nan=False)
        temporary.replace(path)

        print(f"Execution log: {path}")
        return {"status": status, "events": events}

    graph = StateGraph(AgentState)
    graph.add_node("prepare_inputs", prepare_inputs)
    graph.add_node("generate_forecast", generate_forecast)
    graph.add_node("save_log", save_log)

    graph.add_edge(START, "prepare_inputs")
    graph.add_conditional_edges(
        "prepare_inputs",
        lambda state: (
            "stop" if state["status"] == "failed" else "continue"
        ),
        {"stop": "save_log", "continue": "generate_forecast"},
    )
    graph.add_edge("generate_forecast", "save_log")
    graph.add_edge("save_log", END)

    return graph.compile()


def main():
    started = datetime.now(UTC)
    run_id = (
        started.strftime("%Y%m%dT%H%M%S%f")
        + "_" + uuid4().hex[:8]
    )

    initial_state = {
        "run_id": run_id,
        "started_at_utc": started.isoformat(),
        "status": "started",
        "mode": "retrospective_historical_demo",
        "history_path": str(HISTORY_PATH),
        "events": [
            event("workflow", "started", "Historical Forecast Agent started.")
        ],
    }

    print("AgentWeather — LangGraph Forecast Workflow", flush=True)

    try:
        result = build_workflow().invoke(initial_state)
    except Exception as error:
        # Includes failures writing the execution log itself.
        print(f"\nWorkflow could not finish: {type(error).__name__}: {error}")
        return 1

    if result["status"] != "success":
        print(f"\nForecast failed: {result.get('error', 'Unknown error')}")
        return 1

    print("\nForecast workflow completed successfully.")
    print(f"Forecast rows: {result['forecast_rows']}")
    print(f"Forecast CSV: {result['forecast_path']}")
    print("Historical demonstration only—not a current weather forecast.")
    print("No LLM calls, retraining or Hopsworks upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())