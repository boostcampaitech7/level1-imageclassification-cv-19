import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.model import ModelSelector
from src.loss import Loss
from src.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traindata_dir = "/data/ephemeral/home/data/train"
    traindata_info_file = "/data/ephemeral/home/data/train.csv"
    save_result_path = "/data/ephemeral/home/Focal_Loss"

    train_info = pd.read_csv(traindata_info_file)
    num_classes = train_info['target'].nunique()
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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
            np.random.seed(42 + worker_id)
            random.seed(42 + worker_id)
        
        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True,
            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
        )
        
        model_selector = ModelSelector(
            model_type='timm',
            num_classes=num_classes,
            model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
            pretrained=True
        )
        model = model_selector.get_model()
        
        model.to(device)

        # Freeze parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze specific layers
        for name, param in model.named_parameters():
            if 'blocks.23' in name or 'head' in name:
                param.requires_grad = True

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if 'blocks.23' in n and p.requires_grad],
                'lr': 1e-4,  # Intermediate learning rate for specific layers
                'weight_decay': 1e-4
            },
            {
                'params': [p for n, p in model.named_parameters() if 'head' in n and p.requires_grad],
                'lr': 1e-3,  # Higher learning rate for head layers
                'weight_decay': 1e-4
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)

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
