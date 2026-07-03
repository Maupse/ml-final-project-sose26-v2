from pathlib import Path
import pandas as pd
import json

from team_project.helper.parsing import PROJECT_ROOT

def load_raw_listings(path: Path) -> pd.DataFrame:
    """Load the raw listings dataset from a gzip-compressed CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    return pd.read_csv(path, compression="gzip")

def load_single_run_history(experiment_name) -> dict[str, list[float]]:
    path = PROJECT_ROOT / "artifacts/single" / experiment_name / "single_validation_set" / "history.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "Expected a dictionary"
    assert all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(item, float) for item in value)
        for key, value in data.items()
    ), "Expected dict[str, list[float]]"

    return data