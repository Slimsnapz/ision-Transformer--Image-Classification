"""
eval.py
Evaluate a saved ViT model on a test dataset.
Usage:
python eval.py --model_dir models/exp1 --dataset rajistics/indian_food_images
"""
import argparse
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

def compute_metrics(eval_pred):
    accuracy = evaluate.load('accuracy')
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="rajistics/indian_food_images")
    return parser.parse_args()

def main():
    args = parse_args()
    ds = load_dataset(args.dataset)
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)

    from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size['shortest_edge']
        if 'shortest_edge' in image_processor.size
        else (image_processor.size.get('height',224), image_processor.size.get('width',224))
    )
    transforms = Compose([RandomResizedCrop(size if isinstance(size, int) else size[0]), ToTensor(), normalize])

    def transforms_fn(example):
        example["pixel_values"] = transforms(example["image"])
        return example

    ds = ds.with_transform(transforms_fn)

    trainer = Trainer(model=model, tokenizer=image_processor, compute_metrics=compute_metrics)
    results = trainer.predict(ds['test'] if 'test' in ds else ds.get('validation'))
    print("Eval results:", results.metrics)

if __name__ == "__main__":
    # write to disk
    content = open(__file__, "r", encoding="utf-8").read()
    with open("eval.py", "w", encoding="utf-8") as f:
        f.write(content)
    main()
