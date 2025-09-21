# Agent Demo ViT Image Classifier as a LangChain Tool

This demo shows how to wrap the Vision Transformer classifier as a Tool and integrate it with an LLM via LangChain.

## Requirements
- Python 3.9+
- Models: place ViT model at `models/indian_food_vit/`
- Install deps:
  ```bash
  pip install -r requirements.txt
  pip install langchain openai  # if using OpenAI
  # or for local LLM:
  pip install transformers accelerate

