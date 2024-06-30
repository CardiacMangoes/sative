
import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    def __init__(self, folder_path, layer):
        self.folder_path = folder_path
        self.layer = layer

        self.feat_batch_list = []
        for subfolder in os.listdir(folder_path):
            if os.path.isdir(f'{self.folder_path}/{subfolder}'):
                batches = os.listdir(f'{self.folder_path}/{subfolder}')
                batches = [f"{subfolder}/{batch}" for batch in batches]
                self.feat_batch_list.extend(batches)
        
    def __len__(self):
        return len(self.feat_batch_list)
    
    def __getitem__(self, idx):
        feat_batch_name = os.path.join(self.folder_path, self.feat_batch_list[idx])
        with open(feat_batch_name, 'rb') as f:
            feat_batch = np.load(f)[self.layer]
        return feat_batch


class ImageDataset(Dataset):
    def __init__(self, folder_path, processor):
        self.folder_path = folder_path
        self.image_list = os.listdir(folder_path)
        self.processor = processor
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = os.path.splitext(os.path.basename(self.image_list[idx]))[0]
        img_path = os.path.join(self.folder_path, self.image_list[idx])
        image = np.asarray(Image.open(img_path).convert('RGB'))

        pixel_values = self.processor(images=image, return_tensors="pt", padding=False).pixel_values[0]
        
        return img_name, pixel_values