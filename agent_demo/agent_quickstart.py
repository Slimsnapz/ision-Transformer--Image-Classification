"""
agent_quickstart.py
Minimal "agent demo" that integrates the fine-tuned Vision Transformer model
into a LangChain agent-like workflow.

Usage:
    python agent_demo/agent_quickstart.py --image_path examples/idli.jpg
"""

import argparse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI  # or a local LLM wrapper


def load_vit_model(model_dir="models/indian_food_vit"):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def classify_image(image_path, model_dir="models/indian_food_vit", top_k=3):
    processor, model, device = load_vit_model(model_dir)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    ids = probs.argsort()[::-1][:top_k]
    id2label = model.config.id2label
    return [(id2label[i], float(probs[i])) for i in ids]


def main(image_path):
    # Define a LangChain Tool that wraps the ViT classifier
    vit_tool = Tool(
        name="ImageClassifier",
        func=lambda x: str(classify_image(x)),
        description="Classify a food image into one of 20 dish categories."
    )

    # Initialize a simple agent (requires OpenAI key or other LLM backend)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = initialize_agent(
        tools=[vit_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    print("🔎 Asking agent to classify the uploaded image...")
    response = agent.run(f"Classify this food image: {image_path}")
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image for classification")
    args = parser.parse_args()
    main(args.image_path)
