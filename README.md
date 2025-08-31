# **Vision Transformer — Image Classification (Portfolio Project)**

![Project Screenshot](path/to/screenshot.png)  
*Replace the path above with your training/evaluation screenshot or demo image.*

---

## Project Summary (Business Problem)

**Problem:** A fast-growing food delivery company needs to automatically classify images of dishes uploaded by partner restaurants and users. Manually labeling images is slow and error-prone, leading to miscategorized menus, poor search results, and suboptimal recommendations. This project builds a Vision Transformer (ViT) image classification pipeline to automate dish/cuisine classification, improving catalog accuracy, search relevance, and recommendation quality — delivering direct business value in operational efficiency and customer experience.

**My role:** I designed and implemented an end-to-end solution — from dataset preparation and model fine-tuning to deployment as a lightweight Streamlit demo — demonstrating how machine learning can solve a practical business problem.

---

## Key Outcomes & Business Impact

- Automated image classification reduces manual labeling overhead and speeds up onboarding of new restaurant partners.
- Improved menu search & recommendation accuracy → better user experience and potential uplift in conversion.
- Quick validation tool (Streamlit demo) to allow non-technical staff to test and verify model outputs before production rollout.
- Scalable training using Hugging Face `Trainer` for repeatable, auditable experiments.

---

## Technical Highlights

- Model: **Vision Transformer (ViT)** — `google/vit-base-patch16-224-in21k` (fine-tuned).
- Frameworks: Hugging Face `transformers`, `datasets`, PyTorch, `torchvision`, `evaluate`.
- Deployment demo: **Streamlit** app (`app.py`) for non-technical stakeholder demos.
- Reproducibility: training script (`train.py`) with `TrainingArguments`, evaluation script (`eval.py`), and inference script (`predict.py`).

---

## File structure (what to upload to GitHub)

```
/ (root)
├─ notebooks/
│  └─ Vision_Transformers_Image_classification.ipynb   # Original Colab notebook (cleaned)
├─ src/
│  ├─ data_loader.py    # dataset + processor helpers
│  ├─ train.py          # training script (Trainer-based)
│  ├─ eval.py           # evaluation script
│  ├─ predict.py        # single-image prediction script
│  └─ app.py            # Streamlit demo app for non-technical demos
├─ models/              # saved checkpoints (do not commit large binaries)
│  └─ README.md         # instructions for storing checkpoints (Git LFS or cloud)
├─ screenshots/
│  └─ training_loss.png
├─ requirements.txt
├─ README.md            # (this file)
└─ .gitignore
```

**Notes:**
- Don't commit large model checkpoints to the repo. Use Git LFS or a cloud storage link and reference it in `/models/README.md`.
- Place small example images for the demo in `screenshots/demo_images/` if desired.

---

## How this solves the business problem (short)

1. **Ingest images** from partners and customers into a standardized dataset (notebooks show cleaning & transforms).  
2. **Fine-tune a ViT model** with labeled examples so the classifier learns cuisine/dish categories.  
3. **Evaluate** model performance and iterate using metrics (accuracy, confusion matrices).  
4. **Deploy a demo** (Streamlit) for stakeholders to validate the model on real images.  
5. **Operationalize**: integrate predictions into the product pipeline to auto-tag images and enhance search/recommendation features.

---

## Quickstart — Run locally (recommended)

1. Clone the repository
```bash
git clone https://github.com/<YourUsername>/vit-image-classification.git
cd vit-image-classification
```

2. Create virtual environment & install dependencies
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```

3. Train (example)
```bash
python src/train.py --dataset rajistics/indian_food_images --model_ckpt google/vit-base-patch16-224-in21k --output_dir models/exp1
```

4. Evaluate
```bash
python src/eval.py --model_dir models/exp1 --dataset rajistics/indian_food_images
```

5. Predict a single image
```bash
python src/predict.py --model_dir models/exp1 --image_path path/to/image.jpg
```

6. Run Streamlit demo (non-technical stakeholders)
```bash
streamlit run src/app.py -- --model_dir models/exp1
# then open the URL shown in the terminal (usually http://localhost:8501)
```

---

## Requirements (`requirements.txt` suggested content)

```
transformers>=4.30.0
datasets
torch
torchvision
accelerate
evaluate
streamlit
Pillow
scikit-learn
```

Add GPU drivers/CUDA toolkit if training on GPU.

---

## Recommended GitHub workflow and tips

- Create a clean `main` branch with the notebook moved to `notebooks/` and scripts under `src/`.
- Add `.gitignore` to exclude virtual envs, checkpoint files, and large data:
```
__pycache__/
venv/
*.pyc
models/*
!models/README.md
.DS_Store
```
- Use **Git LFS** for model checkpoint artifacts (or store model weights in cloud: S3 / Google Drive / Hugging Face Hub).
- Include `models/README.md` with instructions on where to download trained weights or how to run training to reproduce them.
- Add a short demo video or GIF in `screenshots/` to increase recruiter engagement.

---

## What to include in the repo to impress employers

- Well-documented **README** (this file).  
- Clean `notebooks/` (one runnable notebook or a Colab link).  
- `src/` scripts with CLI flags (so reviewers can run training/eval easily).  
- `requirements.txt` and short **Quickstart** (above).  
- `MODEL_CARD.md` (model purpose, limitations, ethical considerations).  
- Small demo (Streamlit) and screenshots or a short screencast.

---

## Suggested README blurb (short version for repo front page)

> Vision Transformer (ViT) fine-tuned for image classification (food/dish/cuisine). Solves the business problem of automatically tagging and categorizing dish images for a food-delivery platform, reducing manual labeling, improving search and recommendations, and speeding partner onboarding. Includes training, evaluation, inference scripts, and a Streamlit demo.

---

## License & Contact

**License:** MIT (or choose your preferred license)  
**Author:** _Your Name / GitHub handle_  
**Contact:** _your.email@example.com_ | [LinkedIn](https://www.linkedin.com/in/yourprofile)

---

If you'd like, I can:
1. Save this README into `/mnt/data/README.md` so you can download it directly (I already saved it here), or  
2. Create `requirements.txt`, `.gitignore`, `models/README.md`, and `MODEL_CARD.md` files in the folder so you can upload them as a complete package.  
3. Create a zipped project of the `src/` scripts and README for direct upload.

Tell me which of (2) or (3) you want me to produce next and I will create them now.
