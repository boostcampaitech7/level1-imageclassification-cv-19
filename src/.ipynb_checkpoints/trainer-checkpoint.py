import os
import torch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.utils import EarlyStopping

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, epochs, result_path, fold_idx, patience=3, min_delta=0.001):
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
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    def save_model(self, epoch, loss):
        os.makedirs(self.result_path, exist_ok=True)
        model_path = os.path.join(self.result_path, f'fold_{self.fold_idx}_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), model_path)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        scaler = GradScaler()
        for images, targets in tqdm(self.train_loader):
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
        return total_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        return total_loss / len(self.val_loader), 100 * correct / total

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            self.save_model(epoch, val_loss)
            self.scheduler.step()
            if self.early_stopping(val_loss):
                print('Early stopping.')
                break
