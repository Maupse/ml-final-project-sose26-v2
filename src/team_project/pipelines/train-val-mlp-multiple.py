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

from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from itertools import product


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
    preprocessor = get_preprocessor(config)
    X_train = fit_transform(preprocessor, X_train)
    X_val = transform(preprocessor, X_val)
    
    # 2. Run sanity check
    sanity_check("train", X_train, y_train)
    sanity_check("val", X_val, y_val)

    # Input size is number of transformed features
    input_dim = X_train.shape[1]
    output_dim = 1


    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(PROJECT_ROOT / "artifacts" / "multiple" / name / "single_validation_set")
    
    run_id = 1
    for batch_size, epochs, lr, model_type, optimizer_type, loss_fn_type in product(
        config["training"]["batch_size"],
        config["training"]["epochs"],
        config["training"]["lr"],
        config["training"]["model"],
        config["training"]["optimizer"],
        config["training"]["loss_fn"]
    ):
        model = choose_model(model_type, input_dim, output_dim)
        optimizer = choose_optimizer(optimizer_type, model, lr)
        loss_fn = choose_loss_fn(loss_fn_type)
        
        train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
        
        history = train_once(model, optimizer, loss_fn, train_loader, val_loader, epochs, f"run_{run_id}") 

        run_artifact_folder = experiment_artifact_folder / f"run_{run_id:03d}"
        run_artifact_folder.mkdir(parents=True, exist_ok=True)


        history_path = run_artifact_folder / "history.json"
        with history_path.open('w') as f:
            json.dump(history, f, indent=2)

        meta_data_path = run_artifact_folder / "metadata.json"
        with meta_data_path.open('w') as f:
            run_config = config.copy()
            run_config["training"]["batch_size"] = batch_size,
            run_config["training"]["epochs"] = epochs,
            run_config["training"]["lr"] = lr,
            run_config["training"]["model"] = model_type,
            run_config["training"]["optimizer"] = optimizer_type,
            run_config["training"]["loss_fn"] = loss_fn_type
            json.dump(run_config, f, indent=2)
        
        run_id += 1




if __name__ == "__main__":
    main()
