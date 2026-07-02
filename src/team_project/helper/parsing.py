import argparse
import tomllib
from pathlib import Path
import numpy as np
import torch
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config file",
    )
    return parser.parse_args()


def load_config(path: Path):
    with path.open("rb") as file:
        return tomllib.load(file)

def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)