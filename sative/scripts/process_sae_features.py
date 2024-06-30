from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import os
from tqdm import tqdm
import tyro
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader

from sative.datasets import FeatureDataset
from sative.models import SparseAutoencoder

def main():
    in_dim: int = 768
    sae_expansion: int = 16

    target_layer: int = 12
    batch_size = 32768
    num_epochs = 4
    epoch = 4

    model = SparseAutoencoder(in_dim, sae_expansion * in_dim)
    model.load_params(f"checkpoints/{target_layer:02d}_{batch_size:05d}_{num_epochs:03d}/weights_{epoch:03d}")
    model.eval()

    dataset = FeatureDataset("features", layer=target_layer)
    dataloader = DataLoader(dataset, batch_size=2 ** 16, shuffle=False, num_workers=4)

    output_folder = Path("sae_features") / f"{target_layer:02d}_{batch_size:05d}_{num_epochs:03d}/weights_{epoch:03d}"
    output_folder.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(enumerate(dataloader))
    for i, batch in pbar:
        X_hat, f = model(batch)
        np.save(output_folder / f"{i:05d}", f)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()