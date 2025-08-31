"""
predict.py
Load a fine-tuned ViT model and run prediction on a single image or folder.
Usage:
python predict.py --model_dir models/exp1 --image_path path/to/image.jpg
"""
import argparse
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
import numpy as np

def load_model(model_dir):
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return image_processor, model, device

def predict_image(model_dir, image_path):
    image_processor, model, device = load_model(model_dir)
    img = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = int(np.argmax(logits.cpu().numpy(), axis=1)[0])
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    label = id2label.get(pred, str(pred)) if id2label else str(pred)
    return label

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    label = predict_image(args.model_dir, args.image_path)
    print("Prediction:", label)

if __name__ == "__main__":
    # write to disk
    content = open(__file__, "r", encoding="utf-8").read()
    with open("predict.py", "w", encoding="utf-8") as f:
        f.write(content)
    main()
