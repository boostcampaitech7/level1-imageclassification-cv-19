import os
import time
import argparse
import logging
from typing import Tuple, Callable, Optional, Union
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import cv2
import timm
import torch
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm.auto import tqdm  
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
from torch.cuda.amp import GradScaler, autocast  # 추가된 import 문

# Setup logging
logging.basicConfig(
    filename='training.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom Dataset with Stylization and Frequency-Based Augmentations
class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False,
        stylize: bool = False,
        low_freq_aug: bool = False
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.stylize = stylize
        self.low_freq_aug = low_freq_aug
        self.image_paths = info_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def stylize_image(self, image):
        pil_image = Image.fromarray(image)
        pil_image = pil_image.convert("L")  # Convert to grayscale
        grayscale_image = np.array(pil_image)
        image_3_channel = np.stack([grayscale_image] * 3, axis=-1)  # Create 3-channel grayscale image
        return image_3_channel

    def low_pass_filter(self, image):
        pil_image = Image.fromarray(image)
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=2))
        return np.array(blurred_image)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.stylize:
            image = self.stylize_image(image)
        
        if self.low_freq_aug:
            image = self.low_pass_filter(image)
        
        image = self.transform(image)

        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target

# Custom Timm Model with Additional Layers
class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool):
        super(TimmModel, self).__init__()
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # We will add our own classifier head
        )
        
        # Freeze the backbone except for the last few layers (if necessary)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Add custom head: MLP with additional layers
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return {'features': features, 'logits': output}

# Albumentations Transform with Shape and Frequency Bias Focus
class AlbumentationsTransform:
    def __init__(self, img_size: int, is_train: bool = True):
        self.is_train = is_train
        common_transforms = [
            A.Resize(img_size, img_size),  # Use variable image size
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.auto_augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
            self.transform = A.Compose(
                [
                    A.Rotate(limit=15),
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        pil_image = Image.fromarray(image)  # Convert to PIL Image for AutoAugment
        if self.is_train:
            pil_image = self.auto_augment(pil_image)
        
        image = np.array(pil_image)  # Convert back to NumPy array
        transformed = self.transform(image=image)
        return transformed['image']

# Combined Loss (SupConLoss + CrossEntropyLoss)
class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.supcon_loss = SupConLoss(temperature)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, features, logits, targets):
        supcon_loss = self.supcon_loss(features, targets)
        ce_loss = self.cross_entropy_loss(logits, targets)
        combined_loss = self.alpha * supcon_loss + (1 - self.alpha) * ce_loss
        return combined_loss

# SupConLoss implementation
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        logits = torch.div(self.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0)), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits) * mask
        log_prob = torch.log(exp_logits / exp_logits.sum(dim=1, keepdim=True))
        return -(log_prob * mask).sum(dim=1).mean()

# Transform Selector
class TransformSelector:
    def __init__(self, transform_type: str, img_size: int):
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
            self.img_size = img_size
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(img_size=self.img_size, is_train=is_train)
        else:
            raise ValueError("Only 'albumentations' is supported in this setup.")
        return transform

# Model Selector
class ModelSelector:
    def __init__(self, model_type: str, num_classes: int, **kwargs):
        if model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:
        return self.model

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# Trainer Class
class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        loss_fn: Union[nn.Module, CombinedLoss],  # Support CombinedLoss
        epochs: int,
        result_path: str,
        early_stopping: Optional[EarlyStopping] = None
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models = []
        self.lowest_loss = float('inf')
        self.early_stopping = early_stopping
        self.scaler = GradScaler()  # Mixed Precision Scaler

    def save_model(self, epoch, loss):
        os.makedirs(self.result_path, exist_ok=True)

        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")
            logging.info(f"Save {epoch}epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast():  # Mixed precision training
                outputs = self.model(images)
                features = outputs['features']
                logits = outputs['logits']
                loss = self.loss_fn(features, logits, targets)

            self.scaler.scale(loss).backward()  # Scaled gradient
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                features = outputs['features']
                logits = outputs['logits']
                loss = self.loss_fn(features, logits, targets)
                total_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                progress_bar.set_postfix(loss=loss.item())
        
        accuracy = 100 * correct / total
        return total_loss / len(self.val_loader), accuracy

    def train(self) -> None:
        start_time = time.time()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            logging.info(f"Epoch {epoch+1}/{self.epochs}")
            
            epoch_start_time = time.time()
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}% | Time: {epoch_duration:.2f}s")
            logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}% | Time: {epoch_duration:.2f}s")

            self.save_model(epoch, val_loss)
            self.scheduler.step()

            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    logging.info("Early stopping")
                    break
        
        total_training_time = time.time() - start_time
        print(f"Training completed in: {total_training_time:.2f}s")
        logging.info(f"Training completed in: {total_training_time:.2f}s")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train a model using timm and albumentations.')
    parser.add_argument('--img_size', type=int, default=448, help='Image size for resizing (default: 448).')
    parser.add_argument('--model_name', type=str, default='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', help='Timm model name to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64).')
    parser.add_argument('--unfreeze_layers', type=int, default=1, help='Number of layers to unfreeze (default: 1).')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    save_result_path = "./train_supcon"

    train_info = pd.read_csv(traindata_info_file)
    num_classes = len(train_info['target'].unique())

    train_df, val_df = train_test_split(train_info, test_size=0.2, stratify=train_info['target'])

    transform_selector = TransformSelector(transform_type="albumentations", img_size=args.img_size)
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform,
        stylize=True,
        low_freq_aug=True
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model_selector = ModelSelector(
        model_type='timm', 
        num_classes=num_classes,
        model_name=args.model_name, 
        pretrained=True
    )
    model = model_selector.get_model()
    model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    loss_fn = CombinedLoss(temperature=0.07, alpha=0.5)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=50,
        result_path=save_result_path,
        early_stopping=early_stopping
    )

    trainer.train()
