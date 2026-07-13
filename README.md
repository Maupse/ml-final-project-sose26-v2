# ML Final Project

Airbnb price-regression experiments with PyTorch MLPs, scikit-learn tree models, shared preprocessing, and artifact-backed notebooks.

## Install

```bash
uv sync
```

Run commands through `uv` so the project dependencies and package imports are available:

```bash
uv run python -m pytest
```

## Configs

Single-run configs in `configs/single/` use scalar values and produce one fitted model. They are used for MLP training and for the final tree evaluation after hyperparameter selection:

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

MLP multi-run configs in `configs/multiple/` use lists for training values; the corresponding pipeline runs every combination:

```toml
[training]
model = ["mlp"]
optimizer = ["adam"]
loss_fn = ["mse"]
epochs = [25]
lr = [0.001, 0.01]
batch_size = [32, 64, 128]
```

Tree search configs also live in `configs/multiple/`, but their lists describe parameter distributions for `RandomizedSearchCV`. Set `data_split.k` for the number of folds (default: 5) and use `[search]` for search settings:

```toml
[data_split]
train = 0.8
test = 0.2
k = 5

[training]
model = "random-forest"
n_estimators = [200, 500]
max_depth = [-1, 10, 20]
max_features = ["sqrt", 0.5]

[search]
n_iter = 30
n_jobs = -1
```

Supported tree models are `random-forest` and `boosted-tree`. Tree search artifacts record the selected parameters in `best_params.json`; copy these into the matching single config before a final full-train/test run.

## Run Pipelines

```bash
uv run python src/team_project/pipelines/train-val-mlp-single.py --config configs/single/mlp-v1.toml
uv run python src/team_project/pipelines/train-val-mlp-multiple.py --config configs/multiple/mlp-v1-batch-sizes.toml
uv run python src/team_project/pipelines/cross-val-tree-multiple.py --config configs/multiple/random-forest-v1.toml
uv run python src/team_project/pipelines/full-train-test-tree.py --config configs/single/random-forest-v1.toml
```

Outputs are written to:

```text
artifacts/single/<experiment-name>/single_validation_set/
artifacts/multiple/<experiment-name>/single_validation_set/run_*/
artifacts/multiple/<experiment-name>/cross_validation/best_params.json
artifacts/single/<experiment-name>/final/metrics.json
```

MLP runs contain `history.json` and `metadata.json`. Tree searches contain `metadata.json` and `best_params.json`; final tree runs contain `metadata.json` and `metrics.json`.

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
- [x] Add the expanded structural feature set and rerun the model comparison
- [x] Add cross validation to have more stability in testing hyperparameters
- [x] Add random-forest and boosted-tree search and final-evaluation pipelines
- [ ] Evaluate additional feature sets and tune the strongest models further
