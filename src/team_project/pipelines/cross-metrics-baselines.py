from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from team_project.helper.parsing import parse_args, load_config, set_seed, PROJECT_ROOT
from sklearn.model_selection import train_test_split, KFold

from team_project.data.preprocessing import (
    get_preprocessor,
    fit_transform,
    transform,
    sanity_check
)

from team_project.evalutation.metrics import get_metrics_tuple

import pandas as pd
import numpy as np

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

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]

    X = df[features]
    y = df[[target]]
    
    X_train_full, _, y_train_full, _ = \
        train_test_split(X, y, test_size=test_size, random_state=seed)

    preprocessor = get_preprocessor(config)

    
    k = config["data_split"].get("k", 5)
    cv = KFold(
        n_splits=k,
        shuffle=True,
        random_state=seed
    )
    
    
    fold_mse_dummy = []
    fold_mse_linear = []
    fold_rmse_dummy = []
    fold_rmse_linear = []
    fold_mae_dummy = []
    fold_mae_linear = []
    fold_r2_dummy = []
    fold_r2_linear = []

    for (train_idx, val_idx) in cv.split(X_train_full):
        X_train = X_train_full.iloc[train_idx]
        X_val = X_train_full.iloc[val_idx]
        y_train = y_train_full.iloc[train_idx]
        y_val = y_train_full.iloc[val_idx]

        # 1. Fit preprocessor
        X_train = fit_transform(preprocessor, X_train)
        X_val = transform(preprocessor, X_val)
    
        # 2. Run sanity check
        sanity_check("train", X_train, y_train)
        sanity_check("val", X_val, y_val)
        
        dummy = DummyRegressor(strategy="mean")
        linear = LinearRegression()
        
        dummy.fit(X_train, y_train)
        linear.fit(X_train, y_train)
        
        dummy_pred = dummy.predict(X_val)
        linear_pred = linear.predict(X_val)

        mse, rmse, mae, r2 = get_metrics_tuple(y_val, dummy_pred)
        fold_mse_dummy.append(mse)
        fold_rmse_dummy.append(rmse)
        fold_mae_dummy.append(mae)
        fold_r2_dummy.append(r2)

        mse, rmse, mae, r2 = get_metrics_tuple(y_val, linear_pred)
        fold_mse_linear.append(mse)
        fold_rmse_linear.append(rmse)
        fold_mae_linear.append(mae)
        fold_r2_linear.append(r2)
        
    fold_mse_dummy = np.array(fold_mse_dummy)
    fold_mse_linear = np.array(fold_mse_linear)
    fold_rmse_dummy = np.array(fold_rmse_dummy)
    fold_rmse_linear = np.array(fold_rmse_linear)
    fold_mae_dummy = np.array(fold_mae_dummy)
    fold_mae_linear = np.array(fold_mae_linear)
    fold_r2_dummy = np.array(fold_r2_dummy)
    fold_r2_linear = np.array(fold_r2_linear)

    dummy_metrics = {
        "mse": float(fold_mse_dummy.mean()),
        "rmse": float(fold_rmse_dummy.mean()),
        "mae": float(fold_mae_dummy.mean()),
        "r2": float(fold_r2_dummy.mean()),
        "mse_std": float(fold_mse_dummy.std()),
        "rmse_std": float(fold_rmse_dummy.std()),
        "mae_std": float(fold_mae_dummy.std()),
        "r2_std": float(fold_r2_dummy.std()),
    }

    linear_metrics = {
        "mse": float(fold_mse_linear.mean()),
        "rmse": float(fold_rmse_linear.mean()),
        "mae": float(fold_mae_linear.mean()),
        "r2": float(fold_r2_linear.mean()),
        "mse_std": float(fold_mse_linear.std()),
        "rmse_std": float(fold_rmse_linear.std()),
        "mae_std": float(fold_mae_linear.std()),
        "r2_std": float(fold_r2_linear.std()),
    }

    name = config["experiment"]["name"]
    experiment_artifact_folder = Path(PROJECT_ROOT / "artifacts" / "single" / name / "cross_validation")
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
