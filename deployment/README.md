# AgentWeather saved-results dashboard

This deployment displays selected historical and experimental results.
It does not collect live data, run forecasting models, or send alerts.

Experimental results are partial and do not establish reliability.
Reference data are ERA5 reanalysis and CAMS model estimates.

Entrypoint: deployment/dashboard.py
Python version: 3.11

Run locally from the repository root:
python -m streamlit run deployment/dashboard.py

This is a deployment copy. Changes to the root dashboard do not
automatically update this copy or its bundled results.
