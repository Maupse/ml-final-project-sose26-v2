# ML Final Project

Small PyTorch training setup for Airbnb price experiments.

## Install

```bash
uv sync
```

Run commands through `uv` so the project dependencies and package imports are available:

```bash
uv run python -m pytest
```

## Configs

Single-run configs live in `configs/single/`. Use scalar training values:

```toml
[experiment]
name = "mlp-city-center-v1"
seed = 727
dataset = "data/processed/legacy.csv"
features = ["distance_km", "review_scores_rating", "number_of_reviews"]
target = "price"

[data_split]
train = 0.7
validation = 0.1
test = 0.2

[preprocessing]
imputer = "constant"
fill_value = 0

[training]
model = "mlp"
optimizer = "adam"
loss_fn = "mse"
epochs = 20
lr = 0.001
batch_size = 64
```

Multi-run configs live in `configs/multiple/`. Use lists for training values; the pipeline runs every combination:

```toml
[training]
model = ["mlp"]
optimizer = ["adam"]
loss_fn = ["mse"]
epochs = [25]
lr = [0.001, 0.01]
batch_size = [32, 64, 128]
```

## Run Pipelines

```bash
uv run python src/team_project/pipelines/train-val-mlp-single.py --config configs/single/mlp-v1.toml
uv run python src/team_project/pipelines/train-val-mlp-multiple.py --config configs/multiple/mlp-v1-batch-sizes.toml
```

Outputs are written to:

```text
artifacts/single/<experiment-name>/single_validation_set/
artifacts/multiple/<experiment-name>/single_validation_set/run_*/
```

Each run contains `history.json` and `metadata.json`.

## Read Results In A Notebook

```python
from team_project.data.loading import load_data_single_run, load_data_multi_run

single = load_data_single_run("mlp-city-center-v1")
history = single["history"]
metadata = single["metadata"]

runs = load_data_multi_run("mlp-city-center-v1-batch-sizes")
first_run_history = runs[0]["history"]
first_run_metadata = runs[0]["metadata"]
```

Plot a loss curve:

```python
import matplotlib.pyplot as plt

plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.legend()
plt.show()
```

## TODO

- [x] Make train single config work for mlp (Fr, 3. July)
- [x] Add tqdm logging to training loops
- [x] Add train multiple configs and artifacts to be able to test many hyperparameters
- [x] Add linear and mean baseline comparison
- [ ] Work on the expanded feature set, the training data is the current bottleneck
- [x] Add cross validation to have more stability in testing hyperparameters
- [ ] Make pipelines for random forests and boosted trees
