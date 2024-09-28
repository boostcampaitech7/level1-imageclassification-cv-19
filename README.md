# Sketch Image Classification

This repository contains the solution for the **Sketch Image Classification** competition, where the task is to develop a model that can classify objects depicted in sketch images. The primary goal of this project is to build a robust classification model that can recognize abstract shapes and forms, which are often simplified and lack color and texture.

## Project Structure

Below is the tree structure of the repository. The code is organized to easily run training, inference, and model evaluation.

```bash
.
├── data/                    # Dataset directory
│   ├── train/               # Training images
│   ├── test/                # Test images
│   ├── train.csv            # Training labels
│   └── test.csv             # Test file names
├── src/                     # Source code directory
│   ├── dataset.py           # Custom dataset implementation
│   ├── transforms.py        # Data augmentation and transformation logic
│   ├── model.py             # Model selection and initialization
│   ├── loss.py              # Custom loss functions (e.g., Focal Loss)
│   ├── trainer.py           # Training and evaluation logic
│   └── utils.py             # Utility functions
├── train.py                 # Training script
├── inference.py             # Inference script
├── requirements.txt         # Python dependencies
└── README.md                # This readme file
```
## Dataset
![image](https://github.com/user-attachments/assets/c9edb818-1b96-45a6-9ab3-86cea0104aa3)

The dataset used in this competition is derived from the **ImageNet Sketch** dataset, which originally contains 50,889 hand-drawn images across 1,000 classes. For this competition, the dataset has been curated and refined to focus on 500 classes, resulting in 25,035 images. The dataset is split into:

- **Training Data**: 15,021 images
- **Evaluation Data**: 10,014 images (public and private)

Each image represents a simplified, hand-drawn version of an object from one of the 500 categories.

## Model Development

The goal is to develop a model that can learn to recognize and classify objects based on their simplified sketches. The project focuses on capturing the core structure and form of objects without relying on color or texture.

Key steps include:
- Preprocessing the sketch images
- Data augmentation to enhance the model's generalization
- Selecting and fine-tuning appropriate architectures (e.g., Vision Transformers, CNNs)
- Training the model using the training dataset
- Generating predictions for the evaluation dataset

## Evaluation Metric

The performance of the model is evaluated using **Accuracy**, which measures the proportion of correctly predicted images compared to the total number of predictions. The formula for accuracy is:

Accuracy = (Number of Correct Predictions / Total Predictions) * 100

Our goal is to maximize this metric by fine-tuning the model and improving its ability to generalize to unseen sketch images.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.x
- PyTorch
- torchvision
- albumentations
- timm
- pandas
- tqdm
- numpy
- scikit-learn
- PIL

You can install the necessary libraries by running:

```bash
pip install -r requirements.txt
```
## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/sketch-image-classification.git
   cd sketch-image-classification
   ```
2. **Prepare the data**:

Place the training and evaluation images in the appropriate directories (data/train, data/test).
Ensure that the training labels are stored in train.csv and test file names in test.csv.

3. **Train the model**:

You can train the model using the provided script:

```bash
python train.py --epochs 50 --batch-size 32 --lr 1e-4
```

4. **Generate predictions**:

After training, generate predictions for the test set by running:

```bash
python inference.py --model-path saved_model.pth --output predictions.csv
```

5. **Submit results**:

The output.csv file will contain the class probabilities for each image in the evaluation set, ready for submission.

## Results
![image](https://github.com/user-attachments/assets/8775217e-7884-444e-95d3-680fa109cf29)

After training the model on the provided dataset, we achieved an accuracy of **0.9390%** on the evaluation dataset. Our final submission ranked **3rd place** on the competition leaderboard.
