import torch
from team_project.models import (
    mlp,
)

from sklearn.model_selection import train_test_split

def choose_model(model_config, input_dim, output_dim):
    match model_config:
        case "linear":
            return torch.nn.Linear(input_dim, output_dim)
        case "mlp":
            return mlp.MLP(input_dim, output_dim)
        case _:
            raise ValueError("Unknown model in the config: ", model_config)

def choose_optimizer(optimizer_config, model, lr):
    match optimizer_config:
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        case _:
            raise ValueError("Unknown optimizer in the config: ", optimizer_config)



def choose_loss_fn(loss_fn_config):
    match loss_fn_config:
        case "mse":
            return torch.nn.MSELoss()
        case _:
            raise ValueError("Unknown loss_fn in the config: ", loss_fn_config)


def choose_eval_set(X, y, eval_set, config):
    seed = config["experiment"]["seed"]
    if eval_set == "test":
        test_size = config["data_split"]["test"] # 0.20
        X_train_full, X_test, y_train_full, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=seed)
        X_train = X_train_full
        X_eval = X_test
        y_train = y_train_full
        y_eval = y_test
    elif eval_set == "val":
        test_size = config["data_split"]["test"]  # 0.20
        validation_size_within_train_full = (
            config["data_split"]["validation"]
            / (
                config["data_split"]["train"]
                + config["data_split"]["validation"]
            )
        )  # 0.10 / 0.80 = 0.125
        X_train_full, X_test, y_train_full, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=seed)
        X_train, X_val, y_train, y_val = \
            train_test_split(X_train_full, y_train_full, test_size=validation_size_within_train_full, random_state=seed)
        X_eval = X_val
        y_eval = y_val
    else:
        raise ValueError("Wrong --set argument, must be 'val' or 'test'")

    return X_train, X_eval, y_train, y_eval