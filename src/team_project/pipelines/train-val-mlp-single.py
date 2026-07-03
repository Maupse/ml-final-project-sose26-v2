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
    transform
)


import pandas as pd

from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config["experiment"]["seed"]
    set_seed(seed) 
    
    data_path = config["experiment"]["dataset"]
    df = pd.read_csv(PROJECT_ROOT / data_path)
    
    test_size = config["data_split"]["test"]  # 0.20
    validation_size_within_train_full = (
        config["data_split"]["validation"]
        / (
            config["data_split"]["train"]
            + config["data_split"]["validation"]
        )
    )  # 0.10 / 0.80 = 0.125

    features = config["experiment"]["features"]
    target = config["experiment"]["target"]

    X = df[features]
    y = df[[target]]
    
    X_train_full, X_test, y_train_full, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train_full, y_train_full, test_size=validation_size_within_train_full, random_state=seed)

    # 1. Run preprocessing
    imputer = config["preprocessing"]["imputer"]
    fill_value = config["preprocessing"]["fill_value"] # only used if imputer is "constant"

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer, fill_value=fill_value)),
        ("scaler", MinMaxScaler()),        
    ])
    
    X_train = fit_transform(preprocessor, X_train)
    X_val = transform(preprocessor, X_val)
    X_test = transform(preprocessor, X_test)
    
    # 2. Run sanity check
    sanity_check("train", X_train, y_train)
    sanity_check("val", X_val, y_val)
    sanity_check("test", X_test, y_test)

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
    
    # Input size is number of features
    input_dim = len(features)
    output_dim = 1
    model = choose_model(config["training"]["model"], input_dim, output_dim)
    optimizer = choose_optimizer(config["training"]["optimizer"], model, lr)
    loss_fn = choose_loss_fn(config["training"]["loss_fn"])

    history = train_once(model, optimizer, loss_fn, train_loader, val_loader, epochs) 


    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(PROJECT_ROOT / "artifacts" / "single" / name / "single_validation_set")
    experiment_artifact_folder.mkdir(parents=True, exist_ok=True)

    history_path = experiment_artifact_folder / "history.json"
    with history_path.open('w') as f:
        json.dump(history, f, indent=2)

    meta_data_path = experiment_artifact_folder / "metadata.json"
    with meta_data_path.open('w') as f:
        json.dump(config, f, indent=2)




if __name__ == "__main__":
    main()
