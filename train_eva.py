
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
from PIL import Image

# Setup logging
logging.basicConfig(
    filename='training.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Custom Dataset
class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        
        # 이미지가 존재하는지 확인
        if not os.path.exists(img_path):
            # 파일이 없으면 다음 인덱스로 이동 (재귀적으로 호출하여 이미지가 있을 때까지 찾음)
            return self.__getitem__((index + 1) % len(self.image_paths))
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        return output

# Updated Albumentations Transform with CLAHE

# Updated Albumentations Transform with CLAHE and AutoAugment
class AlbumentationsTransform:
    def __init__(self, img_size: int, is_train: bool = True):
        self.is_train = is_train
        common_transforms = [
            A.Resize(img_size, img_size),  # Use variable image size
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.auto_augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)  # Add AutoAugment
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
        
        # Apply torchvision auto augmentation before Albumentations transforms
        pil_image = Image.fromarray(image)  # Convert to PIL Image for AutoAugment
        if self.is_train:
            pil_image = self.auto_augment(pil_image)  # Apply AutoAugment
        
        image = np.array(pil_image)  # Convert back to NumPy array
        transformed = self.transform(image=image)
        return transformed['image']
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
class ModelSelector:
    def __init__(self, model_type: str, num_classes: int, **kwargs):
        if model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:
        return self.model
# Loss Calculation
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets)

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

# Training Class
class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
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
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
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
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                progress_bar.set_postfix(loss=loss.item())
        
        accuracy = 100 * correct / total  # Validation accuracy 계산
        return total_loss / len(self.val_loader), accuracy

    def train(self) -> None:
        start_time = time.time()  # Record training start time
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

            # Early Stopping 체크
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
    parser.add_argument('--img_size', type=int, default=448, help='Image size for resizing (default: 448).')  # Set img_size to 448
    parser.add_argument('--model_name', type=str, default='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', help='Timm model name to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64).')  # Reduced batch size
    parser.add_argument('--unfreeze_layers', type=int, default=1, help='Number of layers to unfreeze (default: 1).')
    args = parser.parse_args()

    # Select device for training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset information.
    traindata_dir = "./dog_remove/data/train"
    traindata_info_file = "./dog_remove/data/train.csv"
    save_result_path = "./train_eva_removedog"

    # Read CSV file containing dataset information.
    train_info = pd.read_csv(traindata_info_file)

    # Determine number of classes.
    num_classes = len(train_info['target'].unique())

    # Split data into training and validation sets.
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target']
    )

    # Define transforms with CLAHE and inverted colors.
    transform_selector = TransformSelector(transform_type="albumentations", img_size=args.img_size)
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # Define datasets.
    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )

    # Define DataLoaders.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # Define model with the updated model name and additional head layers.
    model_selector = ModelSelector(
        model_type='timm', 
        num_classes=num_classes,
        model_name=args.model_name,  # Use model name from argument
        pretrained=True
    )
    model = model_selector.get_model()

    # Move model to the selected device.
    model.to(device)

    # Define optimizer and learning rate for the unfrozen parameters only.
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4  # Lower learning rate for better fine-tuning
    )

    # Scheduler settings
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=50,
        eta_min=1e-6
    )

    # Define loss function.
    loss_fn = Loss()

    # Initialize EarlyStopping.
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Initialize Trainer.
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

    # Start training.
    trainer.train()
