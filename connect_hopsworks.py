import os
from pathlib import Path

import hopsworks
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def main():
    required = (
        "HOPSWORKS_API_KEY",
        "HOPSWORKS_PROJECT",
        "HOPSWORKS_HOST",
    )

    settings = {
        name: os.getenv(name, "").strip()
        for name in required
    }

    missing = [name for name, value in settings.items() if not value]

    if missing:
        print(f"Missing .env settings: {', '.join(missing)}")
        raise SystemExit(1)

    print("Connecting to Hopsworks...")

    project = hopsworks.login(
        host=settings["HOPSWORKS_HOST"],
        port=443,
        project=settings["HOPSWORKS_PROJECT"],
        api_key_value=settings["HOPSWORKS_API_KEY"],
        engine="python",
    )

    feature_store = project.get_feature_store()

    print(f"\nConnected to project: {project.name}")
    print(f"Feature Store: {feature_store.name}")
    print("Hopsworks connection successful!")


if __name__ == "__main__":
    main()