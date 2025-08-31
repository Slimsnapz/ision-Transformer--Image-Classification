# Trained Model — Download Instructions

The trained Vision Transformer (ViT) model for this project (~321 MB) is **not** committed to this repository due to size. It is hosted on Google Drive and available to download:

**Google Drive folder (public):**  
https://drive.google.com/drive/folders/1uRqL_HsX_7a1NSVUug79N7Y6T7YeJchk?usp=sharing

---

## Option 1 — Quick manual download (recommended for non-technical users)

1. Open the Google Drive folder link above in your browser.  
2. Right-click the model file (or the zipped archive if present) and choose **Download**.  
3. Unzip (if zipped) and place the resulting folder in the repository root as:

```
models/indian_food_vit/
  ├─ config.json
  ├─ pytorch_model.bin
  ├─ preprocessor_config.json
  ├─ feature_extractor_config.json
  └─ ... other files
```

After placing files under `models/indian_food_vit`, you can load the model in Python:

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
model = AutoModelForImageClassification.from_pretrained("models/indian_food_vit")
processor = AutoImageProcessor.from_pretrained("models/indian_food_vit")
```

---

## Option 2 — Automated download (if the model was uploaded as a single ZIP file)

If you (or I) uploaded a single ZIP file to the Drive folder (recommended), you can use the `gdown` tool to download by file id. Below is a simple script to do that.

### Steps
1. Install `gdown` (if not already installed):  
```bash
pip install --upgrade gdown
```

2. If you have the Google Drive **file id** for the zipped model (the shareable link will look like `https://drive.google.com/file/d/<FILE_ID>/view?...`), run:

```bash
gdown --id <FILE_ID> -O indian_food_vit.zip
unzip indian_food_vit.zip -d models/indian_food_vit
```

3. Then load as shown above.

---

## Option 3 — Scripted download helper (provided in `scripts/`)

If you'd like to automate the download in one step, use the script `scripts/get_model_from_drive.sh` or `scripts/download_model.py` included in this repo. Example usage:

```bash
# Make script executable
chmod +x scripts/get_model_from_drive.sh

# Download and unzip (replace <FILE_ID> with the actual Drive file id)
./scripts/get_model_from_drive.sh <FILE_ID> indian_food_vit.zip
```

or using the Python helper:

```bash
python scripts/download_model.py --file-id <FILE_ID> --out indian_food_vit.zip
```

---

## Notes & Tips

- If the Drive folder currently contains individual model files (not a zip), please download the entire folder manually via the browser (Drive will create a ZIP for you).
- If you want, I can re-upload the model as a single ZIP and provide the direct `gdown` file-id so users can download via the script.
- For reproducibility, consider hosting the model on Hugging Face Hub in the future (recommended for ML projects).

If you want, I can now:
- (A) Create the ZIP on your Drive and give you the `gdown`-ready file id (I can prepare scripts but I can't access your Drive to upload), **or**
- (B) Generate a short README snippet you can paste into your main `README.md` linking to this models README (I already included the link above).

