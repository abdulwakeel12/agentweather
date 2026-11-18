import os
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))


def main():
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        print("Missing API key. Check your .env file.")
        return

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": 24.8607,
        "lon": 67.0011,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 401:
            print("Key not accepted. Check the key or allow time for activation.")
            return

        if response.status_code != 200:
            print(f"Weather request failed: HTTP {response.status_code}")
            return

        data = response.json()
        observed_at = datetime.fromtimestamp(data["dt"], tz=timezone.utc)

        print("\nKarachi — Current Weather")
        print(f"Observation time (UTC): {observed_at.isoformat()}")
        print(f"Temperature: {data['main']['temp']} °C")
        print(f"Humidity: {data['main']['humidity']} %")
        print(f"Wind speed: {data.get('wind', {}).get('speed', 'N/A')} m/s")
        print(f"Wind direction: {data.get('wind', {}).get('deg', 'N/A')} degrees")
        print(f"Conditions: {data['weather'][0]['description']}")

    except requests.exceptions.RequestException:
        # Avoid printing request URLs, which contain the API key.
        print("Connection failed. Check your internet connection and retry.")
    except (ValueError, KeyError, IndexError, TypeError):
        print("The API returned an unexpected response.")


if __name__ == "__main__":
    main()