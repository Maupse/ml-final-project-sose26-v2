import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def configure_notebook_style() -> None:
    """Apply the project's shared notebook plotting style."""
    plt.rcParams["figure.dpi"] = 120
    sns.set_theme(
        style="ticks",
        font="sans-serif",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
        },
    )


def plot_cv_rmse_comparison(cv_summary: pd.DataFrame) -> None:
    """Plot cross-validated RMSE for a model comparison table."""
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=cv_summary.sort_values("cv_rmse"),
        x="model",
        y="cv_rmse",
        hue="model",
        palette="colorblind",
        legend=False,
    )
    plt.title("Cross-Validated Log-Price RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE on log1_price")
    plt.xticks(rotation=20, ha="right")
    sns.despine()
    plt.show()


def plot_test_error_comparison(
    test_performance: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (9, 5),
) -> None:
    """Plot held-out RMSE and MAE for a model comparison table."""
    test_error_metrics = test_performance.melt(
        id_vars="model",
        value_vars=["rmse", "mae"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=figsize)
    sns.barplot(
        data=test_error_metrics,
        x="metric",
        y="value",
        hue="model",
        palette="colorblind",
    )
    plt.title("Held-Out Test RMSE and MAE on Log Price")
    plt.xlabel("Metric")
    plt.ylabel("Log-price error")
    plt.legend(title="Model")
    sns.despine()
    plt.show()


def plot_test_r2_comparison(
    test_performance: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (8, 4),
) -> None:
    """Plot held-out R2 for a model comparison table."""
    plt.figure(figsize=figsize)
    sns.barplot(
        data=test_performance.sort_values("r2", ascending=False),
        x="model",
        y="r2",
        hue="model",
        palette="colorblind",
        legend=False,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Held-Out Test R2 on Log Price")
    plt.xlabel("Model")
    plt.ylabel("R2")
    plt.xticks(rotation=20, ha="right")
    sns.despine()
    plt.show()

def plot_hyperparameter_runs(
    runs: list[dict],
    hyperparameter_key: str,
    hyperparameter_name: str,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> None:
    """
    Generated using GPT5.5 High
    """
    rows: list[dict] = []

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "Quantiles must satisfy "
            "0 <= lower_quantile < upper_quantile <= 1"
        )

    for run_index, run in enumerate(runs):
        history = run["history"]
        training_config = run["metadata"]["training"]

        hyperparameter_value = training_config[hyperparameter_key]

        if isinstance(hyperparameter_value, list):
            if len(hyperparameter_value) != 1:
                raise ValueError(
                    f"{hyperparameter_key!r} must contain exactly one "
                    f"value per run, got {hyperparameter_value!r}"
                )

            hyperparameter_value = hyperparameter_value[0]

        train_losses = history["train_loss"]
        val_losses = history["val_loss"]

        if len(train_losses) != len(val_losses):
            raise ValueError(
                f"Run {run_index} has different numbers of training "
                "and validation losses"
            )

        for epoch, (train_loss, val_loss) in enumerate(
            zip(train_losses, val_losses),
            start=1,
        ):
            rows.extend(
                [
                    {
                        "run": run_index,
                        "epoch": epoch,
                        "loss": train_loss,
                        "split": "Training",
                        "hyperparameter": hyperparameter_value,
                    },
                    {
                        "run": run_index,
                        "epoch": epoch,
                        "loss": val_loss,
                        "split": "Validation",
                        "hyperparameter": hyperparameter_value,
                    },
                ]
            )

    if not rows:
        raise ValueError("No run data was provided")

    plot_data = pd.DataFrame(rows)

    unique_values = plot_data["hyperparameter"].unique().tolist()

    try:
        unique_values = sorted(unique_values)
    except TypeError:
        unique_values = sorted(unique_values, key=str)

    value_to_label = {
        value: f"{value:g}" if isinstance(value, (int, float)) else str(value)
        for value in unique_values
    }

    plot_data["hyperparameter_label"] = plot_data[
        "hyperparameter"
    ].map(value_to_label)

    hue_order = [
        value_to_label[value]
        for value in unique_values
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.lineplot(
        data=plot_data,
        x="epoch",
        y="loss",
        hue="hyperparameter_label",
        hue_order=hue_order,
        style="split",
        style_order=["Training", "Validation"],
        units="run",
        estimator=None,
        dashes={
            "Training": "",
            "Validation": (4, 2),
        },
        linewidth=2,
        palette="colorblind",
        ax=ax,
    )

    max_epoch = int(plot_data["epoch"].max())
    ax.set_xlim(1, max_epoch)

    tick_step = max(1, max_epoch // 12)
    ticks = list(range(1, max_epoch + 1, tick_step))

    if ticks[-1] != max_epoch:
        ticks.append(max_epoch)

    ax.set_xticks(ticks)

    lower = plot_data["loss"].quantile(lower_quantile)
    upper = plot_data["loss"].quantile(upper_quantile)

    loss_range = upper - lower

    if loss_range > 0:
        padding = 0.05 * loss_range
        ax.set_ylim(lower - padding, upper + padding)

    ax.set_title(
        f"Training and Validation Loss by {hyperparameter_name}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend(
        title=f"{hyperparameter_name} / split",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
