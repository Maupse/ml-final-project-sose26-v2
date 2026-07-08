from pathlib import Path
import pandas as pd
import json

from team_project.helper.parsing import PROJECT_ROOT

def load_raw_listings(path: Path) -> pd.DataFrame:
    """Load the raw listings dataset from a gzip-compressed CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    return pd.read_csv(path, compression="gzip")

def load_data(path) -> dict[str, dict[str, list[float]]]:
    hist = path / "history.json"
    meta = path / "metadata.json"
    data = {}

    with open(hist, "r", encoding="utf-8") as f:
        data["history"] = json.load(f)
    with open(meta, "r", encoding="utf-8") as f:
        data["metadata"] = json.load(f)

    return data

def load_mean_baseline(experiment_name, set):
    path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "single_validation_set" / "baselines")
    file_name = f"mean_{set}_metrics.json"
    file_path = path / file_name
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics

def load_linear_baseline(experiment_name, set):
    path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "single_validation_set" / "baselines")
    file_name = f"linear_{set}_metrics.json"
    file_path = path / file_name
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def load_data_single_run(experiment_name) -> dict[str, dict[str, list[float]]]:
    path = Path(PROJECT_ROOT / "artifacts/single" / experiment_name / "single_validation_set")

    data = load_data(path)

    return data

def load_data_multi_run(experiment_name) -> list[dict[str, dict[str, list[float]]]]:
    parent = Path(PROJECT_ROOT / "artifacts" / "multiple" / experiment_name / "single_validation_set")
    l = []
    for child in parent.iterdir():
        if child.is_dir():
            l.append(load_data(child))

    return l
