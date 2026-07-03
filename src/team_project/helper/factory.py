import torch
from team_project.models import (
    mlp,
)

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

