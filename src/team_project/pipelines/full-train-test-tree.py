from team_project.helper.parsing import (
    parse_args,
    load_config,
    set_seed,
    PROJECT_ROOT,
)

from team_project.helper.factory import choose_tree_model

from team_project.data.preprocessing import (
    sanity_check,
    fit_transform,
    transform,
    get_preprocessor,
)

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from team_project.evalutation.metrics import get_metrics


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
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    preprocessor = get_preprocessor(config)

    X_train_full = fit_transform(preprocessor, X_train_full)
    X_test = transform(preprocessor, X_test)

    sanity_check("train", X_train_full, y_train_full)
    sanity_check("test", X_test, y_test)

    model = choose_tree_model(config)
    model.fit(X_train_full, y_train_full)

    pred = model.predict(X_test)
    metrics = get_metrics(y_test, pred)

    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(PROJECT_ROOT / "artifacts" / "single" / name / "final")
    experiment_artifact_folder.mkdir(parents=True, exist_ok=True)

    meta_data_path = experiment_artifact_folder / "metadata.json"
    with meta_data_path.open("w") as f:
        json.dump(config, f, indent=2)

    metrics_path = experiment_artifact_folder / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
