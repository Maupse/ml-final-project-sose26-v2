import pandas as pd


def summarize_cross_validation_runs(
    runs: list[dict],
    hyperparameter_key: str,
    hyperparameter_name: str,
) -> pd.DataFrame:
    """Summarize the final and best validation loss for MLP CV runs."""
    rows = []

    for run in runs:
        training = run["metadata"]["training"]
        history = run["history"]
        validation_losses = history["val_loss"]

        rows.append(
            {
                hyperparameter_name: training[hyperparameter_key],
                "final_val_mse": validation_losses[-1],
                "final_val_mse_std": history["val_loss_std"][-1],
                "best_epoch": pd.Series(validation_losses).idxmin() + 1,
                "best_val_mse": min(validation_losses),
            }
        )

    return pd.DataFrame(rows).sort_values("final_val_mse")
