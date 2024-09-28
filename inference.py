import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from argparse import ArgumentParser
from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.model import ModelSelector
from src.trainer import set_seed
import torch.nn.functional as F

def tta_inference(models, device, test_loader, base_transform, tta_transform, tta_steps=3):
    if not models:
        raise ValueError("No models provided for inference.")
    
    for model in models:
        model.to(device)
        model.eval()

    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="TTA Inference"):
            B = len(images)
            base_images = torch.stack([base_transform(image) for image in images]).to(device)
            sum_probs = torch.zeros(B, num_classes).to(device)

            for model in models:
                logits = model(base_images)
                probs = F.softmax(logits, dim=1)
                sum_probs += probs

                for _ in range(tta_steps):
                    tta_images = torch.stack([tta_transform(image) for image in images]).to(device)
                    logits = model(tta_images)
                    probs = F.softmax(logits, dim=1)
                    sum_probs += probs / tta_steps

            avg_probs = sum_probs / len(models)
            preds = avg_probs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

if __name__ == "__main__":
    # Argparse for command line arguments
    parser = ArgumentParser()
    parser.add_argument('--testdata_dir', type=str, default='/data/ephemeral/home/dog_remove/data/test')
    parser.add_argument('--testdata_info_file', type=str, default='/data/ephemeral/home/dog_remove/data/test.csv')
    parser.add_argument('--save_result_path', type=str, default='/data/ephemeral/home/youngtae/Focal_Loss')
    parser.add_argument('--model_paths', nargs='+', default=[f"/data/ephemeral/home/youngtae/Focal_Loss/fold_{i}_best_model.pt" for i in range(1, 6)])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tta_steps', type=int, default=3)
    args = parser.parse_args()

    set_seed(42)

    # Load test data
    test_info = pd.read_csv(args.testdata_info_file)
    num_classes = len(test_info['target'].unique())

    test_dataset = CustomDataset(
        root_dir=args.testdata_dir,
        info_df=test_info,
        transform=None,
        is_inference=True
    )

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=lambda x: x
    )

    # Set up transforms
    transform_selector = TransformSelector(transform_type='torchvision')
    base_transform = transform_selector.get_transform(is_train=False)
    tta_transform = transform_selector.get_transform(is_train=True)

    # Load models
    models = []
    for model_path in args.model_paths:
        model_selector = ModelSelector(
            model_type='timm',
            num_classes=num_classes,
            model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
            pretrained=False
        )
        model = model_selector.get_model()
        model.load_state_dict(torch.load(model_path))
        models.append(model)

    # Perform inference with TTA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = tta_inference(
        models=models,
        device=device,
        test_loader=test_loader,
        base_transform=base_transform,
        tta_transform=tta_transform,
        tta_steps=args.tta_steps
    )

    # Save predictions
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    output_csv_path = os.path.join(args.save_result_path, "output.csv")
    test_info.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
