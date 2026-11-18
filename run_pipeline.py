import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Collect weather", "main.py"),
    ("Collect modelled air quality", "check_air_quality.py"),
    ("Build hourly dataset", "build_hourly_dataset.py"),
]


def main():
    started_at = datetime.now(UTC)
    print(
        f"\nAgentWeather pipeline started: {started_at.isoformat()}",
        flush=True,
    )

    # Check all scripts exist before running any step.
    for _, filename in STEPS:
        if not (BASE_DIR / filename).is_file():
            print(f"Missing script: {filename}", flush=True)
            return 1

    for number, (description, filename) in enumerate(STEPS, start=1):
        print(
            f"\nStep {number}/{len(STEPS)}: {description}",
            flush=True,
        )

        try:
            result = subprocess.run(
                [sys.executable, "-u", str(BASE_DIR / filename)],
                cwd=BASE_DIR,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"\nPipeline stopped: {filename} timed out.", flush=True)
            return 1
        except OSError:
            print(f"\nPipeline stopped: could not run {filename}.", flush=True)
            return 1

        if result.returncode != 0:
            print(
                f"\nPipeline stopped: {filename} failed "
                f"with exit code {result.returncode}.",
                flush=True,
            )
            return 1

        print(f"Step {number} completed.", flush=True)

    elapsed = (datetime.now(UTC) - started_at).total_seconds()

    print("\nPipeline completed successfully.", flush=True)
    print(f"Duration: {elapsed:.1f} seconds", flush=True)
    print(f"Dataset: {BASE_DIR / 'data' / 'hourly_dataset.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())