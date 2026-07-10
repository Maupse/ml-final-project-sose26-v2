from team_project.helper.parsing import (
    parse_args,
    load_config,
    set_seed,
    PROJECT_ROOT
)

from team_project.helper.factory import (
    choose_model,
    choose_optimizer,
    choose_loss_fn,
)

from team_project.training.pytorch_loops import (
    make_loader,
    train_once
)

from team_project.data.preprocessing import (
    sanity_check,
    fit_transform,
    transform,
    get_preprocessor
)


import pandas as pd
import numpy as np

from pathlib import Path
import json

from sklearn.model_selection import train_test_split, KFold
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
    input_dim = len(features)
    output_dim = 1

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]

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
    sanity_check("val", X_test, y_test)

    # 3. Create data loaders
    train_loader = make_loader(X_train_full, y_train_full, batch_size, shuffle=True)
    val_loader = make_loader(X_test, y_test, batch_size, shuffle=False)
    
    # 4. Choose model
    model = choose_model(config["training"]["model"], input_dim, output_dim)
    optimizer = choose_optimizer(config["training"]["optimizer"], model, lr)
    loss_fn = choose_loss_fn(config["training"]["loss_fn"])

    # Train model
    history = train_once(model, optimizer, loss_fn, train_loader, val_loader, epochs, "run-once") 
    
    pred = model(y_test)
    metrics = get_metrics(y_test, pred)


    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(PROJECT_ROOT / "artifacts" / "single" / name / "final")
    experiment_artifact_folder.mkdir(parents=True, exist_ok=True)

    history_path = experiment_artifact_folder / "history.json"
    with history_path.open('w') as f:
        json.dump(history, f, indent=2)

    meta_data_path = experiment_artifact_folder / "metadata.json"
    with meta_data_path.open('w') as f:
        json.dump(config, f, indent=2)

    metrics_path = experiment_artifact_folder / "metrics.json"
    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2)
    

if __name__ == "__main__":
    main()
