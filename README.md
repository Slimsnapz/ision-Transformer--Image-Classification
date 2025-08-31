<img width="1477" height="1307" alt="image" src="https://github.com/user-attachments/assets/f6d4b8a2-e7ba-449b-85a4-62660593b1cd" /># Vision Transformer (ViT) — Food Image Classification

![Project Screenshot](https://github.com/Slimsnapz/ision-Transformer--Image-Classification/blob/cb819e4dff293d8bc29619158cb3541ff7c24d8e/screenshots/Screenshot%202025-08-31%20091742.png) 


*Model training loss.*



## Project summary
I built an end-to-end image classification pipeline that fine-tunes a **Vision Transformer (ViT)** model to classify food images into 20 dish categories. This project was implemented to solve a practical business problem: automatically tagging and categorizing dish images uploaded by restaurants and users for a food-delivery platform. The solution improves catalog accuracy, enhances search and recommendations, and reduces manual labeling effort.

This repository contains the training and evaluation code, a lightweight Streamlit demo for quick model inspection, and helper scripts to download the trained model weights (hosted externally due to size).



## What I did (technical summary)
- **Dataset:** Used `rajistics/indian_food_images` from Hugging Face (20 classes, images per class limited).
- **Preprocessing & Augmentation:** Standard ViT-compatible transforms, normalized with the model processor means/std. Performed RandomResizedCrop and conversion to pixel tensors before feeding into the model.
- **Model:** Fine-tuned `google/vit-base-patch16-224-in21k` using Hugging Face `Trainer` and `AutoModelForImageClassification`.
- **Training setup (used in experiments):**
  - Learning rate: `5e-5`
  - Batch size: `16` (train), `4` (eval)
  - Gradient accumulation: `4`
  - Epochs: `4`
  - Evaluation: `eval per epoch`, save best model by `accuracy`
  - Compute metrics: accuracy (primary). Confusion matrix and per-class review performed in notebooks.
- **Evaluation:** Model evaluation performed on the dataset test/validation split with accuracy and class-level inspection (confusion matrices). *(Replace the values below with the final metrics from your run.)*
  - Example (placeholder): **Validation accuracy:** `0.88` — **Test accuracy:** `0.86`
- **Deployment / Demo:** Built a Streamlit demo (`src/app.py`) allowing stakeholders to upload images and view model predictions (top-k labels and probabilities).



## Business impact (why this matters)
- **Operational efficiency:** Automatically tags dish images during ingestion, removing the need for manual labeling pipelines.
- **Product quality:** Better search and recommendation signals (correct categorization) improve user experience and conversion.
- **Scalability:** The solution is trained via widely-adopted frameworks (Hugging Face / PyTorch) and can be retrained as new labeled data arrives.



## Where the trained model lives
The trained model (~321 MB) is **not** tracked in this repository. It is hosted on Google Drive and can be downloaded using the instructions in `models/README.md` or using the helper script in `scripts/`. Public Drive folder:

https://drive.google.com/drive/folders/1uRqL_HsX_7a1NSVUug79N7Y6T7YeJchk?usp=sharing

After downloading and placing the model at `models/indian_food_vit/`, load it like this:

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
model = AutoModelForImageClassification.from_pretrained("models/indian_food_vit")
processor = AutoImageProcessor.from_pretrained("models/indian_food_vit")
```



## File structure (what's included in the repo)
```
/ (root)
├─ notebooks/
│  └─ Vision_Transformers_Image_classification.ipynb   # original Colab notebook (cleaned) — training & EDA
├─ src/
│  ├─ data_loader.py    # dataset & processor helpers
│  ├─ train.py          # training script (Trainer-based)
│  ├─ eval.py           # evaluation script
│  ├─ predict.py        # single-image prediction script
│  └─ app.py            # Streamlit demo app for stakeholders
├─ scripts/
│  ├─ get_model_from_drive.sh   # bash helper to download zipped model via gdown
│  └─ download_model.py         # python helper to download & unzip model
├─ models/              # NOT committed (large files) — include models/README.md (tracked)
│  └─ README.md         # instructions for downloading trained weights (tracked)
├─ screenshots/
│  └─ training_loss.png
├─ requirements.txt
├─ .gitignore
├─ README.md            # (this file)
└─ LICENSE
```



## Quickstart — run the demo locally (minimal steps)
1. Clone the repo:
```bash
git clone https://github.com/<YourUsername>/vit-image-classification.git
cd vit-image-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

3. Download the trained model and place at `models/indian_food_vit/` (see `models/README.md` or use `scripts/download_model.py`).

4. Run a single-image prediction:
```bash
python src/predict.py --model_dir models/indian_food_vit --image_path path/to/image.jpg
```

5. Run the Streamlit demo:
```bash
streamlit run src/app.py -- --model_dir models/indian_food_vit
# Open the local URL shown in the terminal (usually http://localhost:8501)
```

6. To retrain from scratch (if you want to reproduce training):
```bash
python src/train.py --dataset rajistics/indian_food_images --model_ckpt google/vit-base-patch16-224-in21k --output_dir models/exp1
```

This will fine-tune the ViT model using the training configuration described above. Training time depends on your hardware (GPU recommended).


## Notes, caveats & next steps
- **Data balance & augmentation:** Since the dataset has relatively few images per class, consider additional augmentations or semi-supervised strategies for production-grade performance.
- **Model size & inference:** ViT-base is moderately large; for CPU-first deployments consider quantization or a smaller backbone.
- **Evaluation:** Add per-class precision/recall and a calibration analysis before production deployment.
- **Model hosting:** For easier distribution, consider uploading the model to Hugging Face Hub in the future.


## Contact
If you review this project and want to discuss details, reproduce the results, or see a live demo, reach out: **slimsnaps@gmail.com**

---

**License:** MIT
