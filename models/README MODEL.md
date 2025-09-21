# Trained Model — Download Instructions

The trained Vision Transformer (ViT) model for this project (~321 MB) is **not** committed to this repository due to size. It is hosted on Google Drive and available to download:

**Google Drive folder (public):**  
https://drive.google.com/drive/folders/1uRqL_HsX_7a1NSVUug79N7Y6T7YeJchk?usp=sharing



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



## Option 2 — Scripted download helper (provided in `scripts/`)

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

