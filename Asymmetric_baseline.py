import os
from typing import Tuple, Callable, Union, List
import random

import cv2
import timm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import albumentations as A
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from albumentations.pytorch import ToTensorV2
from PIL import Image


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable = None,  
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

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], Image.Image]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.is_inference:
            return image  #
        else:
            image = self.transform(image)
            target = self.targets[index]
            return image, target


# Torchvision Transform
class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        
        if is_train:
            self.transform = transforms.Compose(
                [
                    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # AutoAugment during training
                ] + common_transforms
            )
        else:
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


# Albumentations Transform
class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(448, 448),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.RandomBrightnessContrast(p=0.2),
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        transformed = self.transform(image=image)
        return transformed['image']


# Transform Selector
class TransformSelector:
    def __init__(self, transform_type: str):
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train)
        elif self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train)
        return transform


# Timm Model Class
class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Model Selector
class ModelSelector:
    def __init__(self, model_type: str, num_classes: int, **kwargs):
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:
        return self.model


# Focal Loss Class
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha]).float()
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha).float()
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha[targets]
        else:
            alpha = 1.0

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()

        loss = -alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=4.9, gamma_neg=4.7, clip=0.05, eps=1e-8, reduction='mean'):
        """
        Args:
            gamma_pos (float): 양성 클래스에 대한 초점 맞추기 파라미터.
            gamma_neg (float): 음성 클래스에 대한 초점 맞추기 파라미터.
            clip (float): 음성 예측에 대한 임계값.
            eps (float): 로그 함수의 안정성을 위한 작은 값.
            reduction (str): 손실의 합산 방식 ('mean' | 'sum' | 'none').
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 모델의 출력 logits (batch_size, num_classes).
            targets: 실제 정답 라벨 (batch_size).
        """
        # 소프트맥스 확률 계산
        probs = F.softmax(inputs, dim=1)  # (batch_size, num_classes)
        
        # 정답 클래스에 대한 확률 추출
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))  # (batch_size, num_classes)
        targets_one_hot = targets_one_hot.type_as(probs)
        p_t = (probs * targets_one_hot).sum(dim=1)  # (batch_size)
        
        # 양성 클래스 손실 계산
        loss_pos = -((1 - p_t) ** self.gamma_pos) * torch.log(p_t + self.eps)
        
        # 음성 클래스 손실 계산
        p_n = (probs * (1 - targets_one_hot)).sum(dim=1)  # (batch_size)
        loss_neg = -((p_n) ** self.gamma_neg) * torch.log(1 - p_n + self.eps)
        
        # 비대칭 손실 합산
        loss = loss_pos + loss_neg

        # 손실 합산 방식
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
# Loss Class
class Loss(nn.Module):
    def __init__(self, loss_type='focal'):
        super(Loss, self).__init__()
        if loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(gamma=2, alpha=None, reduction='mean')
        elif loss_type == 'asymmetric':
            self.loss_fn = AsymmetricLoss(gamma_pos=1, gamma_neg=4, clip=0.05, reduction='mean')
        else:
            raise ValueError("Unsupported loss type.")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Trainer Class
from torch.cuda.amp import autocast, GradScaler

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
        fold_idx: int,
        patience: int = 3, 
        min_delta: float = 0.001
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
        self.fold_idx = fold_idx
        self.best_models = []
        self.lowest_loss = float('inf')
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    def save_model(self, epoch, loss):
        os.makedirs(self.result_path, exist_ok=True)

        current_model_path = os.path.join(
            self.result_path, 
            f'fold_{self.fold_idx}_epoch_{epoch}_loss_{loss:.4f}.pt'
        )
        torch.save(self.model.state_dict(), current_model_path)

        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, f'fold_{self.fold_idx}_best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Fold {self.fold_idx}: Save {epoch} epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        scaler = GradScaler()

        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            progress_bar.set_postfix(loss=loss.item())

        train_accuracy = 100.0 * correct / total
        return total_loss / len(self.train_loader), train_accuracy

    def validate(self) -> tuple:
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

                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                progress_bar.set_postfix(loss=loss.item())

        val_accuracy = 100.0 * correct / total
        return total_loss / len(self.val_loader), val_accuracy

    def train(self) -> None:
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate()

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

            self.save_model(epoch, val_loss)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered. Stopping training...")
                break

            self.scheduler.step()


# Training Script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traindata_dir = "/data/ephemeral/home/data/train"
traindata_info_file = "/data/ephemeral/home/data/train.csv"
save_result_path = "/data/ephemeral/home/Focal_Loss"

train_info = pd.read_csv(traindata_info_file)
num_classes = len(train_info['target'].unique())

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

for fold_idx, (train_index, val_index) in enumerate(skf.split(train_info, train_info['target']), 1):
    print(f'Fold {fold_idx}')

    train_df = train_info.iloc[train_index].reset_index(drop=True)
    val_df = train_info.iloc[val_index].reset_index(drop=True)

    transform_selector = TransformSelector(transform_type="torchvision")
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

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

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    model_selector = ModelSelector(
        model_type='timm', 
        num_classes=num_classes,
        model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', 
        pretrained=True
    )
    model = model_selector.get_model()

    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'blocks.23' in name or 'head' in name:
            param.requires_grad = True

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if 'blocks.23' in n and p.requires_grad],
            'lr': 1e-4,  # blocks.23 레이어는 중간 학습률
            'weight_decay': 1e-4
        },
        {
            'params': [p for n, p in model.named_parameters() if 'head' in n and p.requires_grad],
            'lr': 1e-3,  # head 레이어는 높은 학습률
            'weight_decay': 1e-4
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=1,
        eta_min=1e-6
    )

    # Use Asymmetric Loss
    loss_fn = Loss(loss_type='asymmetric')

    fold_result_path = os.path.join(save_result_path, f'fold_{fold_idx}')
    os.makedirs(fold_result_path, exist_ok=True)

    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=100,
        result_path=fold_result_path,
        fold_idx=fold_idx
    )

    trainer.train()
