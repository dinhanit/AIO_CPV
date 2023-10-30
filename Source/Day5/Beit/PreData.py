import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from datasets import load_dataset
from config import path_dataset
def load_transforms():
    train_transforms = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ]
    )
    return train_transforms, val_transforms



# Load transforms dữ liệu
train_transforms, val_transforms = load_transforms()

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [
        train_transforms(image = np.array(image))["image"] for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [
        val_transforms(image = np.array(image))["image"] for image in example_batch["image"]
        ]
    return example_batch

dataset = load_dataset(path_dataset)
train_ds = dataset["train"]
test_ds = dataset["test"]
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label