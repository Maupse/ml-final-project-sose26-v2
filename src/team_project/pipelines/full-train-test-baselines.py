from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from team_project.helper.parsing import parse_args, load_config, set_seed, PROJECT_ROOT

from team_project.data.preprocessing import (
    get_preprocessor,
    fit_transform,
    transform,
    sanity_check
)

from team_project.evalutation.metrics import get_metrics

import pandas as pd

from pathlib import Path
import json


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config["experiment"]["seed"]
    set_seed(seed)

    data_path = config["experiment"]["dataset"]
    df = pd.read_csv(PROJECT_ROOT / data_path)

    test_size = config["data_split"]["test"]

    features = config["experiment"]["features"]
    target = config["experiment"]["target"]

    X = df[features]
    y = df[[target]]

    X_train_full, X_test, y_train_full, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)

    preprocessor = get_preprocessor(config)

    # 1. Fit preprocessor
    X_train_full = fit_transform(preprocessor, X_train_full)
    X_test = transform(preprocessor, X_test)

    # 2. Run sanity check
    sanity_check("train", X_train_full, y_train_full)
    sanity_check("test", X_test, y_test)

    dummy = DummyRegressor(strategy="mean")
    linear = LinearRegression()

    dummy.fit(X_train_full, y_train_full)
    linear.fit(X_train_full, y_train_full)

    dummy_pred = dummy.predict(X_test)
    linear_pred = linear.predict(X_test)

    dummy_metrics = get_metrics(y_test, dummy_pred)
    linear_metrics = get_metrics(y_test, linear_pred)

    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(PROJECT_ROOT / "artifacts" / "single" / name / "final")
    baseline_folder = experiment_artifact_folder / "baselines"
    baseline_folder.mkdir(parents=True, exist_ok=True)

    dummy_metrics_path = baseline_folder / "mean_metrics.json"
    with dummy_metrics_path.open('w') as f:
        json.dump(dummy_metrics, f, indent=2)

    linear_metrics_path = baseline_folder / "linear_metrics.json"
    with linear_metrics_path.open('w') as f:
        json.dump(linear_metrics, f, indent=2)


if __name__ == "__main__":
    main()
