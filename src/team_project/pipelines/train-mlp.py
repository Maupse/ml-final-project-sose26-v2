from team_project.helper.parsing import (
    parse_args,
    load_config,
    set_seed,
    PROJECT_ROOT
)

from team_project.training.pytorch_loops import (
    make_loader
)

import pandas as pd

from sklearn.model_selection import train_test_split

def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config["experiment"]["seed"]
    set_seed(seed) 
    
    data_path = config["experiment"]["data_path"]
    df = pd.load_csv(PROJECT_ROOT / data_path)
    
    test_size = config["data_split"]["test"]  # 0.20
    validation_size_within_train_full = (
        config["data_split"]["validation"]
        / (
            config["data_split"]["train"]
            + config["data_split"]["validation"]
        )
    )  # 0.10 / 0.80 = 0.125

    features = config["training"]["features"]
    target = config["training"]["target"]

    X = df[features]
    y = df[[target]]

    X_train_full, X_test, y_train_full, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train_full, y_train_full, test_size=validation_size_within_train_full, random_state=seed)


    batch_size = config["training"]["batch_size"]
    
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)



    name = config["experiment"]["name"]
    output_path = config["experiment"]["output_path"]




if __name__ == "__main__":
    main()
