from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        i_dim: int, 
        sae_dim: int
    ):  
        # set up as per https://transformer-circuits.pub/2023/monosemantic-features
        # a bit hacky but I'm not sure of a proper way to initialize weights and biases
        super().__init__()
        self.encoder = nn.Linear(i_dim, sae_dim)
        self.encoder.weight[:] = nn.init.he_uniform()(self.encoder.weight)
        self.encoder.bias[:] = 0

        self.decoder = nn.Linear(sae_dim, i_dim)
        self.decoder.weight[:] = nn.init.he_uniform()(self.decoder.weight)
        self.decoder.weight[:] /= mx.linalg.norm(self.decoder.weight, ord=2, axis=0, keepdims=True) # normalize column weights
        self.decoder.bias[:] = 0

    def __call__(self, x):
        x_bar = x - self.decoder.bias
        f = mx.maximum(self.encoder(x_bar), 0.0)
        x_hat = self.decoder(f)
        return x_hat, f
        
    def save_params(self, out_path: Path):
        out_path.mkdir(parents=True, exist_ok=True)

        np.save(out_path / "encoder_weight", self.encoder.weight)
        np.save(out_path / "encoder_bias", self.encoder.bias)
        np.save(out_path / "decoder_weight", self.decoder.weight)
        np.save(out_path / "decoder_bias", self.decoder.bias)

    def load_params(self, file_path: Path):
        file_path = Path(file_path)
        encoder_weight = np.load(file_path / "encoder_weight.npy")
        encoder_bias = np.load(file_path / "encoder_bias.npy")
        decoder_weight = np.load(file_path / "decoder_weight.npy")
        decoder_bias = np.load(file_path / "decoder_bias.npy")

        self.encoder.weight[:] = encoder_weight
        self.encoder.bias[:] = encoder_bias
        self.decoder.weight[:] = decoder_weight
        self.decoder.bias[:] = decoder_bias