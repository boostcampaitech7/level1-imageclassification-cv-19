import os
import cv2
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Callable, Union

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, info_df: pd.DataFrame, transform: Callable = None, is_inference: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], Image.Image]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.is_inference:
            return image
        else:
            image = self.transform(image)
            target = self.targets[index]
            return image, target
