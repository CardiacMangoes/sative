import wandb
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import tyro
import numpy as np
from mlx.utils import tree_flatten

from sative.datasets import FeatureDataset
from sative.models import SparseAutoencoder

def loss_fn(model, X, l1_coeff=0.0008):
    X_hat, f = model(X)
    mse_loss = mx.mean(mx.sum(mx.power(X_hat - X, 2), 1))
    l1_loss = mx.mean(mx.sum(mx.abs(f), 1))
    return mse_loss + l1_coeff * l1_loss


def main(in_dim: int = 768,
         sae_expansion: int = 16,
         num_epochs: int = 4,
         lr: int = 0.0004,
         batch_size: int = 32768,
         target_layer: int = 12):
    
    wandb.init(project="sparse-autoencoder", config={
        "in_dim": in_dim,
        "sae_expansion": sae_expansion,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "target_layer": target_layer
    })

    dataset = FeatureDataset("features", layer=target_layer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = optim.Adam(learning_rate=lr)

    output_folder = Path("checkpoints") / f"{target_layer:02d}_{batch_size:05d}_{num_epochs:03d}"
    output_folder.mkdir(parents=True, exist_ok=True)

    model = SparseAutoencoder(in_dim, in_dim * sae_expansion)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(num_epochs):
        print(f"epoch {epoch}")
        model.save_params(output_folder / f"weights_{epoch:03d}")
        pbar = tqdm(dataloader)
        for batch in pbar:
            loss, grads = loss_and_grad_fn(model, batch)

            # Update the model with the gradients. So far no computation has happened.
            optimizer.update(model, grads)

            # Compute the new parameters but also the optimizer state.
            mx.eval(model.parameters(), optimizer.state)

            # Log loss to wandb
            wandb.log({"epoch": epoch, "loss": loss.item()})

            pbar.set_description(f"Loss: {loss.item():.2e}", refresh=True)

    model.save_params(output_folder / f"weights_{num_epochs:03d}")
    wandb.finish()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()