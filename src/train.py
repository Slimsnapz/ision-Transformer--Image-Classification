"""
train.py
Train script for fine-tuning Vision Transformer for image classification.
Usage:
python train.py --dataset rajistics/indian_food_images --model_ckpt google/vit-base-patch16-224-in21k --output_dir models/exp1
"""

import argparse
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def compute_metrics(eval_pred):
    accuracy = evaluate.load('accuracy')
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rajistics/indian_food_images")
    parser.add_argument("--model_ckpt", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--output_dir", type=str, default="models/exp1")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    ds = load_dataset(args.dataset)
    labels = ds['train'].features['label'].names
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for i,l in enumerate(labels)}

    # Image processor
    image_processor = AutoImageProcessor.from_pretrained(args.model_ckpt, use_fast=True)

    # Build transforms
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size['shortest_edge']
        if 'shortest_edge' in image_processor.size
        else (image_processor.size.get('height',224), image_processor.size.get('width',224))
    )
    from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
    transforms = Compose([
        RandomResizedCrop(size if isinstance(size, int) else size[0]),
        ToTensor(),
        normalize,
    ])

    def transforms_fn(example):
        example["pixel_values"] = transforms(example["image"])
        return example

    ds = ds.with_transform(transforms_fn)

    # Model
    model = AutoModelForImageClassification.from_pretrained(
        args.model_ckpt,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'] if 'test' in ds else ds.get('validation', None),
        tokenizer=image_processor,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    # write this script file to disk
    content = open(__file__, "r", encoding="utf-8").read()
    with open("train.py", "w", encoding="utf-8") as f:
        f.write(content)
    main()
