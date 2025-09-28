"""
agent_indian_food_detection_app.py

- Run local ViT image classifier (loads from local model dir).
- Use a local HF LLM (e.g., google/flan-t5-small) to produce a 1-2 sentence
  human-readable summary of the classifier predictions.
- Robustness:
  - Do not include raw file paths in prompts (they confuse LLMs).
  - Use few-shot examples and explicit instructions.
  - Validate LLM output and deterministically fallback if the LLM "echoes" a single label.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

# ---------- CONFIG ----------
DEFAULT_MODEL_DIR = r"C:\Users\001\Documents\workbench\indian_food_classification"
DEFAULT_HF_LLM = "google/flan-t5-small"
DEFAULT_TOP_K = 3

# ---------- Helpers ----------
def clean_path(p: str) -> str:
    if not p:
        return p
    p = p.strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    p = p.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    p = os.path.expanduser(p)
    return os.path.abspath(p)

# ---------- Vision model loader & inference ----------
def load_vit_model(model_dir: str):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"ViT model directory not found: {model_dir}")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device

def classify_image(image_path: str, model_dir: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
    processor, model, device = load_vit_model(model_dir)
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"File at {image_path} is not a valid image.")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    ids = probs.argsort()[::-1][:top_k]
    id2label = getattr(model.config, "id2label", {i: str(i) for i in range(len(probs))})
    return [(id2label[int(i)], float(probs[int(i)])) for i in ids]

def print_predictions(preds: List[Tuple[str, float]]):
    print("Top predictions (label, prob):")
    for label, p in preds:
        print(f" - {label}: {p:.2%}")

# ---------- Deterministic fallback summary ----------
def deterministic_summary(preds: List[Tuple[str, float]]) -> str:
    top_label, top_prob = preds[0]
    second = preds[1] if len(preds) > 1 else (None, 0.0)
    second_label, second_prob = second
    top_pct = top_prob * 100
    second_pct = second_prob * 100 if second_label else None

    if top_prob >= 0.85:
        return f"The image very likely shows {top_label} ({top_pct:.1f}% confidence)."
    if second_label and second_prob > 0.10:
        return (f"The image is most likely {top_label} ({top_pct:.1f}% confidence), "
                f"with {second_label} as the next most likely ({second_pct:.1f}%).")
    return f"The image is most likely {top_label} ({top_pct:.1f}% confidence)."

# ---------- Local HF LLM summarizer (improved prompt & validation) ----------
def load_local_hf_llm(model_name: str = DEFAULT_HF_LLM):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading HF LLM '{model_name}' (device={device}) — this may read from cache.")
    pipe = pipeline("text2text-generation", model=model_name, device=device)
    return pipe

def build_prompt_from_preds(preds: List[Tuple[str, float]]) -> str:
    # Build a clean predictions block (no file paths)
    lines = [f"- {label}: {prob*100:.2f}%" for label, prob in preds]
    pred_block = "\n".join(lines)

    # Few-shot examples to guide flan-t5 style models
    examples = (
        "Examples:\n"
        "Predictions:\n- cat: 80.00%\n- dog: 20.00%\nSummary: The image most likely contains a cat (80.0% confidence); "
        "the model shows less support for a dog.\n\n"
        "Predictions:\n- apple: 55.00%\n- orange: 45.00%\nSummary: The image is most likely an apple (55.0% confidence), but the model is fairly uncertain.\n\n"
    )

    instruction = (
        "You are a concise assistant. Given the classifier predictions below, write a short (1-2 sentence) "
        "natural-language summary describing the most likely label and its confidence. "
        "Do NOT repeat the list or print raw label: percentage lines. Keep the summary plain and human-readable.\n\n"
    )

    prompt = instruction + examples + "Predictions:\n" + pred_block + "\n\nSummary:"
    return prompt

def summarize_with_hf(pipe, preds: List[Tuple[str, float]], max_new_tokens: int = 60) -> str:
    prompt = build_prompt_from_preds(preds)
    try:
        # Use max_new_tokens (preferred) to avoid max_length conflicts
        output_list = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        # pipeline returns list of dicts like [{'generated_text': '...'}] depending on transformers version
        if isinstance(output_list, list) and output_list:
            generated = output_list[0].get("generated_text") or output_list[0].get("text") or str(output_list[0])
            text = generated.strip()
        else:
            text = str(output_list).strip()
    except Exception as e:
        return f"[HF summarization failed: {e}]"

    # Validation: avoid one-line echo like "kaathi_rolls: 5.16%"
    short_or_echo = False
    t = text.strip()
    if len(t) < 20:
        short_or_echo = True
    # crude: if entire text looks like "label: xx.xx%" (single-line pattern)
    import re
    if re.fullmatch(r"^[A-Za-z0-9_\- ]+:\s*\d{1,3}(\.\d+)?%$", t):
        short_or_echo = True

    if short_or_echo:
        # fallback deterministic summary
        return deterministic_summary(preds)

    return text

# ---------- Optional OpenAI summarizer (kept for completeness) ----------
def summarize_with_openai(openai_key: str, preds: List[Tuple[str, float]]) -> str:
    try:
        import openai
    except Exception as e:
        return f"[OpenAI python package not installed: {e}]"

    lines = [f"- {label}: {prob*100:.2f}%" for label, prob in preds]
    pred_block = "\n".join(lines)
    prompt = (
        "You are a concise assistant. Given classifier predictions below, write a short (1-2 sentence) summary "
        "stating the most likely label and its confidence.\n\n"
        f"Predictions:\n{pred_block}\n\nSummary:"
    )
    try:
        openai.api_key = openai_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenAI summarization failed: {e}]"

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Image classifier + robust local LLM summarizer.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image to classify")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Local ViT model directory")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of top predictions to return")
    parser.add_argument("--use_local", action="store_true", help="Use local HuggingFace LLM for explanation")
    parser.add_argument("--hf_model", type=str, default=DEFAULT_HF_LLM, help="HF model name for local LLM (text2text)")
    parser.add_argument("--openai_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="Optional OpenAI API key")
    parser.add_argument("--no_summary", action="store_true", help="Don't call any LLM summarizer (just classify)")
    args = parser.parse_args()

    image_path = clean_path(args.image_path)
    model_dir = clean_path(args.model_dir)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path!r}")
        return

    try:
        preds = classify_image(image_path=image_path, model_dir=model_dir, top_k=args.top_k)
    except Exception as e:
        print(f"Image classification failed: {e}")
        return

    print_predictions(preds)

    if args.no_summary:
        print("\n(no LLM summary requested; exiting)")
        return

    # Prefer local HF if requested; else prefer OpenAI if key provided; else attempt local
    summary = None
    provider = None

    if args.use_local:
        try:
            pipe = load_local_hf_llm(args.hf_model)
            provider = f"local:{args.hf_model}"
            summary = summarize_with_hf(pipe, preds)
        except Exception as e:
            print(f"Local HF LLM failed to load: {e}")
            summary = None

    if summary is None and args.openai_key:
        provider = "openai:gpt-3.5-turbo"
        summary = summarize_with_openai(args.openai_key, preds)

    if summary is None:
        # final attempt: try local HF if not tried
        try:
            pipe = load_local_hf_llm(args.hf_model)
            provider = f"local:{args.hf_model}"
            summary = summarize_with_hf(pipe, preds)
        except Exception as e:
            summary = f"[No summarizer available: {e}]"

    print("\n--- LLM Summary ---")
    print(f"(provider: {provider or 'none'})")
    print(summary or "[empty response]")

if __name__ == "__main__":
    main()
