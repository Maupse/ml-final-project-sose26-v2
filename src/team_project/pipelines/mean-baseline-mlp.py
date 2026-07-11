from team_project.helper.parsing import (
    parse_args_baseline,
    load_config,
    set_seed,
    PROJECT_ROOT
)

from team_project.data.preprocessing import (
    sanity_check,
    fit_transform,
    transform,
    get_preprocessor
)

from team_project.evalutation.metrics import get_metrics

from team_project.helper.factory import choose_eval_set

import pandas as pd

from pathlib import Path
import json

from sklearn.dummy import DummyRegressor

def main():
    args = parse_args_baseline()
    config = load_config(args.config)
    eval_set = args.set

    seed = config["experiment"]["seed"]
    set_seed(seed) 
    
    data_path = config["experiment"]["dataset"]
    df = pd.read_csv(PROJECT_ROOT / data_path)
    
    features = config["experiment"]["features"]
    target = config["experiment"]["target"]

    X = df[features]
    y = df[[target]]
    
    X_train, X_eval, y_train, y_eval = choose_eval_set(X, y, eval_set, config)

    # 1. Run preprocessing
    preprocessor = get_preprocessor(config)
    X_train = fit_transform(preprocessor, X_train)
    X_eval = transform(preprocessor, X_eval)

    # 2. Run sanity check
    sanity_check("train", X_train, y_train)
    sanity_check("eval", X_eval, y_eval)

    model = DummyRegressor(strategy="mean")

    model.fit(X_train, y_train)      # stores mean of y_train
    y_pred = model.predict(X_eval)   # always predicts that mean

    
    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(PROJECT_ROOT / "artifacts" / "single" / name / "single_validation_set")
    baseline_folder = experiment_artifact_folder / "baselines"

    baseline_folder.mkdir(parents=True, exist_ok=True)
    metrics_path = baseline_folder / f"mean_{eval_set}_metrics.json"

    metrics = get_metrics(y_eval, y_pred)
    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2)



if __name__ == "__main__":
    main()
