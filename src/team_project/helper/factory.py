import torch
from team_project.models import (
    mlp,
)

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split


TREE_MODEL_PARAMETERS = {
    "random-forest": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
        "n_jobs",
    },
    "boosted-tree": {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "loss",
    },
}


def choose_model(model_config, input_dim, output_dim):
    match model_config:
        case "linear":
            return torch.nn.Linear(input_dim, output_dim)
        case "mlp":
            return mlp.MLP(input_dim, output_dim)
        case _:
            raise ValueError("Unknown model in the config: ", model_config)


def _normalize_tree_param(name, value):
    if name == "max_depth" and value == -1:
        return None
    return value


def _tree_kwargs(config):
    training_config = config["training"]
    model_type = training_config["model"]
    supported_parameters = TREE_MODEL_PARAMETERS[model_type]

    return {
        name: _normalize_tree_param(name, value)
        for name, value in training_config.items()
        if name in supported_parameters
    }


def choose_tree_model(config):
    seed = config["experiment"]["seed"]
    model_type = config["training"]["model"]
    kwargs = _tree_kwargs(config)

    match model_type:
        case "random-forest":
            return RandomForestRegressor(random_state=seed, **kwargs)
        case "boosted-tree":
            return GradientBoostingRegressor(random_state=seed, **kwargs)
        case _:
            raise ValueError("Unknown tree model in the config: ", model_type)


def choose_tree_param_distributions(config, prefix=""):
    model_type = config["training"]["model"]
    if model_type not in TREE_MODEL_PARAMETERS:
        raise ValueError("Unknown tree model in the config: ", model_type)

    supported_parameters = TREE_MODEL_PARAMETERS[model_type]
    return {
        f"{prefix}{name}": [
            _normalize_tree_param(name, value)
            for value in values
        ]
        for name, values in config["training"].items()
        if name in supported_parameters
    }


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
