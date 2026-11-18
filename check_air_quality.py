import requests


def main():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "current": "us_aqi,pm2_5,pm10,uv_index",
        "domains": "cams_global",
        "timezone": "UTC",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException:
        print("Connection failed. Check your internet and retry.")
        return

    if response.status_code != 200:
        print(f"Request failed: HTTP {response.status_code}")
        return

    try:
        data = response.json()
        current = data["current"]
        units = data.get("current_units", {})

        print("\nKarachi — Modelled Air Quality")
        print("Source: Open-Meteo / CAMS Global")
        print("Data type: Model estimate, not a station measurement")
        print(f"Grid coordinates: {data['latitude']}, {data['longitude']}")
        print(f"Valid time (UTC): {current['time']}")

        variables = {
            "us_aqi": "US AQI",
            "pm2_5": "PM2.5 concentration",
            "pm10": "PM10 concentration",
            "uv_index": "UV Index",
        }

        for variable, label in variables.items():
            value = current.get(variable)

            if value is None:
                print(f"{label}: unavailable")
            else:
                unit = units.get(variable, "")
                print(f"{label}: {value} {unit}")

    except (ValueError, KeyError, TypeError, AttributeError):
        print("The API returned an unexpected response.")


if __name__ == "__main__":
    main()