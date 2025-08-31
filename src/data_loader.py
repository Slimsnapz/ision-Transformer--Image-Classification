"""
data_loader.py
Data loading utilities for Vision Transformer image classification project.
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import os

def load_image_dataset(dataset_name="rajistics/indian_food_images"):
    """
    Loads a Hugging Face dataset containing images.
    Default uses 'rajistics/indian_food_images' as in the notebook.
    """
    ds = load_dataset(dataset_name)
    return ds

def build_image_processor(model_ckpt="google/vit-base-patch16-224-in21k"):
    """
    Returns an AutoImageProcessor for the model checkpoint.
    """
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)
    return image_processor

def build_transforms(image_processor):
    """
    Build torchvision transforms compatible with image_processor.
    """
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size['shortest_edge']
        if 'shortest_edge' in image_processor.size
        else (image_processor.size.get('height',224), image_processor.size.get('width',224))
    )

    transforms = Compose([
        RandomResizedCrop(size if isinstance(size, int) else size[0]),
        ToTensor(),
        normalize,
    ])
    return transforms

if __name__ == "__main__":
    # write this helper file to disk for convenience
    content = open(__file__, "r", encoding="utf-8").read()
    with open("data_loader.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Wrote data_loader.py")
