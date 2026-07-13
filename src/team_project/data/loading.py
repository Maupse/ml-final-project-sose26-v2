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


def load_cross_validation_mean_baseline(experiment_name):
    path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "cross_validation" / "baselines")
    file_path = path / "mean_metrics.json"
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def load_cross_validation_linear_baseline(experiment_name):
    path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "cross_validation" / "baselines")
    file_path = path / "linear_metrics.json"
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def load_final_metrics(experiment_name):
    file_path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "final" / "metrics.json")
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def load_tree_best_params(experiment_name):
    """Load the best parameters selected by a tree-model search."""
    file_path = Path(
        PROJECT_ROOT
        / "artifacts"
        / "multiple"
        / experiment_name
        / "cross_validation"
        / "best_params.json"
    )
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tree_search_metadata(experiment_name):
    """Load metadata produced by a tree-model randomized search."""
    file_path = Path(
        PROJECT_ROOT
        / "artifacts"
        / "multiple"
        / experiment_name
        / "cross_validation"
        / "metadata.json"
    )
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_final_mean_baseline(experiment_name):
    path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "final" / "baselines")
    file_path = path / "mean_metrics.json"
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def load_final_linear_baseline(experiment_name):
    path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "final" / "baselines")
    file_path = path / "linear_metrics.json"
    if not file_path.is_file():
        raise FileNotFoundError("The artifact might have not been created yet")
    with open(file_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def load_data_single_run(experiment_name) -> dict[str, dict[str, list[float]]]:
    path = Path(PROJECT_ROOT / "artifacts/single" / experiment_name / "single_validation_set")

    data = load_data(path)

    return data


def load_data_single_run_cross_validation(experiment_name) -> dict[str, dict[str, list[float]]]:
    path = Path(PROJECT_ROOT / "artifacts" / "single" / experiment_name / "cross_validation")

    data = load_data(path)

    return data


def load_data_multi_run(experiment_name) -> list[dict[str, dict[str, list[float]]]]:
    parent = Path(PROJECT_ROOT / "artifacts" / "multiple" / experiment_name / "single_validation_set")
    runs = []
    for child in sorted(parent.iterdir()):
        if child.is_dir():
            runs.append(load_data(child))

    return runs


def load_data_multi_run_cross_validation(experiment_name) -> list[dict[str, dict[str, list[float]]]]:
    parent = Path(PROJECT_ROOT / "artifacts" / "multiple" / experiment_name / "cross_validation")
    runs = []
    for child in sorted(parent.iterdir()):
        if child.is_dir():
            runs.append(load_data(child))

    return runs
