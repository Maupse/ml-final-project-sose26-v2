from team_project.helper.parsing import (
    parse_args,
    load_config,
    set_seed,
    PROJECT_ROOT,
)

from team_project.helper.factory import (
    choose_tree_model,
    choose_tree_param_distributions,
)

from team_project.data.preprocessing import get_preprocessor

import copy
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline


def _strip_pipeline_prefix(params):
    return {
        key.removeprefix("model__"): value
        for key, value in params.items()
    }


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

    X_train_full, _X_test, y_train_full, _y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    k = config["data_split"].get("k", 5)
    search_config = config.get("search", {})

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor(config)),
        ("model", choose_tree_model(config)),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=choose_tree_param_distributions(config, prefix="model__"),
        n_iter=search_config.get("n_iter", 10),
        scoring=search_config.get("scoring", "neg_root_mean_squared_error"),
        cv=k,
        random_state=seed,
        n_jobs=search_config.get("n_jobs"),
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train_full, y_train_full)

    best_params = _strip_pipeline_prefix(search.best_params_)

    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(
        PROJECT_ROOT / "artifacts" / "multiple" / name / "cross_validation"
    )
    experiment_artifact_folder.mkdir(parents=True, exist_ok=True)

    best_params_path = experiment_artifact_folder / "best_params.json"
    with best_params_path.open("w") as f:
        json.dump(best_params, f, indent=2)

    metadata_path = experiment_artifact_folder / "metadata.json"
    with metadata_path.open("w") as f:
        run_config = copy.deepcopy(config)
        run_config["data_split"]["k"] = k
        run_config["search"]["best_score"] = search.best_score_
        run_config["search"]["best_params"] = best_params
        json.dump(run_config, f, indent=2)


if __name__ == "__main__":
    main()
