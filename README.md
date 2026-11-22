# AgentWeather

A Karachi weather and air-quality forecasting project combining machine learning, statistical forecasting, deep learning, LangGraph workflows, Hopsworks data storage, and a Streamlit dashboard.

The system produces hourly predictions for the next 24 forecast horizons.

> **Current status:** The deployed dashboard displays saved historical and experimental results. It does not automatically collect fresh inputs or generate live forecasts. Accuracy with the experimental GFS weather inputs has not been established.

## Dashboard

**Deployed app:** [https://abdulwakeel12-agentweather-deploymentdashboard-vkwvrm.streamlit.app/]

The dashboard contains four sections:

- **Weather Overview:** predicted values and hourly charts.
- **Alerts and Report:** historical alert windows and factual summaries; experimental alerts are disabled.
- **Model Performance:** historical test results and available experimental comparisons with baselines.
- **Historical Analysis:** training-data distributions, daily trends, and hourly patterns.

Users can switch between historical and experimental results. Refreshing the dashboard reloads saved files; it does not run models or call weather APIs.

## Forecast outputs

| Output | Unit or representation |
|---|---|
| US AQI | Modelled US AQI |
| Temperature | °C |
| Precipitation | mm per hour |
| Humidity | % |
| UV Index | Index |
| Wind speed | m/s |
| Wind direction | Degrees, derived from predicted sine and cosine |
| Rain probability | Probability of rain ≥0.1 mm in an individual target hour |

Hourly rain probabilities are not the probability of rain at any time during the entire day. Probability calibration remains unestablished.

Wind-direction predictions are generated, but their dashboard display is deferred until a policy for calm or ambiguous predictions is defined.

## Data sources

| Source | Role | Data type |
|---|---|---|
| Open-Meteo / ERA5 | Historical weather training and evaluation reference | Reanalysis |
| Open-Meteo / CAMS Global | AQI, particulate matter and UV inputs/reference values | Model estimates |
| OpenWeather | Separate current-weather collection pipeline | Provider-reported current weather |
| Open-Meteo / GFS Global | Experimental recent weather inputs | Weather-model estimates |
| Hopsworks | Storage for collected weather and air-quality records | Feature groups |

CAMS values are not measurements from a Karachi monitoring station. ERA5 is a reanalysis reference, not independent station-observed truth.

The current-weather collection pipeline and the experimental GFS input pipeline are separate. Missing rainfall in collected OpenWeather records is preserved rather than silently converted to zero.

## Historical dataset and features

The historical dataset covers **1 November 2025 through 31 October 2026**, with 8,760 hourly rows.

The feature builder creates **147 features**, including:

- Current values.
- Lags of 1, 3, 6, 24, 48 and 168 hours.
- Rolling means and standard deviations over preceding 3, 6 and 24 hours.
- One-hour changes.
- Circular wind-direction features.
- Cyclical calendar features using Karachi time.

Lag and rolling features use earlier timestamps. Current-value features require the input to be available when a forecast is generated.

The first 168 hours provide feature warm-up. Training examples whose future targets cross a split boundary are excluded.

| Split | Calendar period | Usable supervised prediction times |
|---|---|---:|
| Training | November 2025–July 2026 | 6,360 |
| Validation | August–September 2026 | 1,440 |
| Test | October 2026 | 720 |

Forecast targets cover horizons **+1 through +24 hours**.

## Models

The project compared a previous-day baseline, Ridge regression, Random Forest, SARIMAX and LSTM models.

Continuous-output models were selected using mean validation RMSE across the 24 horizons. Rain probabilities were compared using Brier scores, and wind direction using angular errors.

| Output | Selected model |
|---|---|
| US AQI | Random Forest |
| Temperature | SARIMAX |
| Precipitation | LSTM |
| Humidity | LSTM |
| UV Index | Random Forest |
| Wind speed | LSTM |
| Rain probability | Random Forest classifiers |
| Wind direction | Ridge models predicting sine and cosine |

Saved models were reloaded and checked against their saved validation predictions. This verified prediction reproduction within numerical tolerances, not forecast accuracy.

## October test results

Models were selected before evaluating the October test period. The following values are **mean RMSE across the 24 horizons**, in each output's units.

| Output | Selected model | Previous-day baseline |
|---|---:|---:|
| US AQI | 9.1117 | 14.9094 |
| Temperature | 1.0366 | 1.0226 |
| Precipitation | 0.0113 | 0.0149 |
| Humidity | 10.9751 | 10.4268 |
| UV Index | 0.4249 | 0.3980 |
| Wind speed | 1.2241 | 1.2934 |

Results were mixed:

- AQI and wind speed improved on the previous-day baseline on average.
- Temperature, humidity and UV did not improve on that baseline.
- Precipitation improved on the previous-day baseline but did not beat the zero-precipitation baseline, whose mean RMSE was approximately 0.0105.
- Performance varies by horizon; an average improvement does not imply improvement at every horizon.

Additional results:

- Rain-probability mean Brier score: **0.011018**, compared with **0.011111** for always dry.
- At +24 hours and threshold 0.5, the rain classifier detected none of the eight rainy target hours.
- Wind-direction mean angular MAE: **26.21°**, compared with **28.32°** for the previous-day baseline.

These results do not establish consistent superiority over simple baselines or accuracy against independent station observations.

## LangGraph workflow

The historical workflow connects four components:

1. **Input monitoring:** checks completeness and screens distribution changes.
2. **Forecasting:** loads saved models and produces 24 hourly predictions.
3. **Alert evaluation:** identifies historical AQI and UV alert windows.
4. **Report generation:** produces a factual weather summary.

Execution logs connect the outputs of each run.

Reports currently use a fixed template, not an LLM. Distribution screening does not measure forecast-error drift. Automatic retraining and external notifications are not implemented.

## Experimental recent-input workflow

The experimental pipeline:

1. Downloads and validates a 169-hour GFS/CAMS snapshot.
2. Combines it with preserved historical inputs.
3. Rejects missing hours instead of filling gaps.
4. Rebuilds the existing feature representation.
5. Verifies that SARIMAX training temperatures remain unchanged.
6. Generates predictions using fixed saved models.
7. Saves persistence and previous-day baselines.
8. Saves the exact input history, metadata and execution log.
9. Updates a local history checkpoint after successful forecasting.

SARIMAX needs its historical state, so the recent snapshot alone is insufficient.

### Source compatibility

The models were trained using ERA5 weather, while experimental recent weather inputs use GFS.

A comparison of 39 overlapping hours found average absolute differences of approximately:

- Temperature: **1.64 °C**
- Humidity: **19.28 percentage points**
- Wind speed: **1.13 m/s**
- Wind direction: **37.87°**

Both sources were dry during that comparison, so rainfall compatibility was not established. These are source differences, not errors against observed truth.

No bias correction was fitted from this short comparison.

### Experimental evaluation

The evaluator compares saved model predictions and baselines with available ERA5/CAMS reference values.

It:

- Preserves original predictions.
- Excludes intervals that began before forecast generation.
- Scores only elapsed eligible targets.
- Leaves missing references pending.
- Uses matching eligible rows across compared methods.
- Saves separate evaluation reports with input-file hashes.

Partial scores from one run do not establish reliability. Repeated evaluation of the same forecast is not a new independent forecast sample.

## Technology

- Python 3.11
- pandas and NumPy
- scikit-learn
- TensorFlow / Keras
- statsmodels
- LangGraph
- Hopsworks
- Streamlit and Plotly
- GitHub Actions

## Main files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Collect weather and air-quality inputs |
| `build_hourly_dataset.py` | Construct the collected hourly dataset |
| `upload_weather.py` | Upload weather records to Hopsworks |
| `upload_air_quality.py` | Upload air-quality records to Hopsworks |
| `build_features.py` | Build historical or inference features |
| `build_targets.py` | Build future targets |
| `train_models.py` | Train classical regression candidates |
| `train_statistical.py` | Train statistical forecasting candidates |
| `train_lstm.py` | Train LSTM candidates |
| `train_rain_probability.py` | Train rain-event probability candidates |
| `train_wind_direction.py` | Train circular wind-direction candidates |
| `evaluate_final_test.py` | Evaluate frozen selections on the test period |
| `forecast.py` | Generate historical or experimental predictions |
| `agent_workflow.py` | Run the historical LangGraph workflow |
| `alert_agent.py` | Evaluate historical alert windows |
| `report_agent.py` | Generate factual reports |
| `drift_agent.py` | Check historical input quality and distributions |
| `collect_recent_inputs.py` | Save recent GFS/CAMS snapshots |
| `run_experimental_pipeline.py` | Collect inputs and generate experimental forecasts |
| `evaluate_experimental.py` | Evaluate a saved experimental run |
| `dashboard.py` | Explore local saved results |
| `deployment/dashboard.py` | Deployed saved-results demonstration |

Generated datasets, trained models and most local artifacts are excluded from Git. The deployment folder contains only selected demonstration outputs.

## Run the packaged dashboard

The packaged dashboard is the simplest way to explore the project after cloning the repository.

```powershell
git clone https://github.com/abdulwakeel12/agentweather.git
cd agentweather

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r deployment/requirements.txt
python -m streamlit run deployment/dashboard.py
```

No API credentials or trained model files are needed for this saved-results dashboard.

## Full local pipeline

Running the complete forecasting pipeline requires local historical datasets, feature files, selected model artifacts and the evaluation plan. These are not all included in the repository.

The root requirements file contains the recorded collection and training dependencies. LangGraph and dashboard dependencies must also be available for their respective components.

```powershell
python -m pip install -r requirements.txt
python -m pip install langgraph
python -m pip install -r deployment/requirements.txt
python -m pip check
```

The following commands assume the required datasets and saved models already exist:

```powershell
# Historical workflow
python agent_workflow.py

# Collect recent inputs, forecast and save baselines
python run_experimental_pipeline.py

# Evaluate one saved experimental run
python evaluate_experimental.py --run "data/experimental_forecasts/RUN_ID"

# Explore local saved results
python -m streamlit run dashboard.py
```

Replace `RUN_ID` with an actual run folder name.

Some scripts contain project-specific dataset dates and artifact paths. Review those settings before attempting to rebuild the project with a different dataset.

## Credentials

The separate collection/upload workflow uses:

- `OPENWEATHER_API_KEY`
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT`
- `HOPSWORKS_HOST`

Keep credentials in local environment variables or GitHub secrets/settings as appropriate. Do not commit `.env` files or credentials.

The deployed saved-results dashboard requires no API secrets.

## Automation and deployment

GitHub Actions is configured to collect weather and air-quality records and upload them to Hopsworks.

This collection workflow does not automatically run the experimental forecasting pipeline or update the deployed dashboard.

The Streamlit deployment uses:

- Entrypoint: `deployment/dashboard.py`
- Dependencies: `deployment/requirements.txt`
- Bundled results: `deployment/demo_data/`

The deployment is a separate saved-results package. Changes to the root dashboard, local forecasts or evaluations do not automatically update this package.

## Limitations and remaining work

- Validate recent-input forecasts across more periods and weather conditions.
- Assess the ERA5-to-GFS source change more thoroughly.
- Improve rainfall evaluation with more rain events.
- Establish probability calibration and an alert-threshold policy.
- Connect scheduled forecasting to durable artifact storage and dashboard updates.
- Add forecast-error monitoring.
- Define a controlled retraining and model-promotion process.
- Add LLM reporting if required by the project specification.
- Consolidate full-pipeline dependencies and improve reproducibility.

The current system is a forecasting research prototype and deployed demonstration. It should not be presented as a validated operational warning service.

## Data attribution

- [Open-Meteo](https://open-meteo.com/)
- [ERA5 / Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/)
- [OpenWeather](https://openweathermap.org/)

Weather and air-quality outputs should retain their source and data-type labels.