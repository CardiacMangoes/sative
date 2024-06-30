# Extracts the residual stream of the class token for a set of images at each encoder layer
# clip-vit-base-patch32 has 12 transformer encoder layers so we extract 12 tokens

import tyro
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

from sative.datasets import ImageDataset
    
def load_image(filename):
    image = np.asarray(Image.open(filename).convert('RGB'))
    return image

def main(data: str):
    if torch.backends.mps.is_available():
        device = "mps" 
    else:
        device = "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    dataset_name = data.split('/')[-1]
    output_folder = Path("features") / dataset_name
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset = ImageDataset(data, processor)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            img_names, pixel_values = batch
            vision_outputs = model.vision_model(
                        pixel_values=pixel_values.to(device),
                        output_attentions=False,
                        output_hidden_states=True,
                    )
            hidden_states = vision_outputs.hidden_states
            hidden_class_token_states = torch.stack([state[:,0,:] for state in hidden_states], dim=1).cpu()
            for img_name, hidden_class_token_state in zip(img_names, hidden_class_token_states):
                with open(output_folder / f"{img_name}.npy", 'wb') as f:
                    np.save(f, hidden_class_token_state.numpy())
    

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()