# Vision Transformer (ViT) — Food Image Classification
### APP DEMO 
![Project Screenshot](https://github.com/Slimsnapz/ision-Transformer--Image-Classification/blob/5c4523264ca1366eaf94d58cb56f853244673797/screenshots/Screenshot%202025-09-21%20072458.png)
### APP AFTER LOADING INDIAN FOOD IMAGE CLASSIFICATION MODEL
![Project Screenshot](https://github.com/Slimsnapz/ision-Transformer--Image-Classification/blob/5c4523264ca1366eaf94d58cb56f853244673797/screenshots/Screenshot%202025-09-21%20072537.png)
### APP AFTER LOADING IMAGE TO BE CLASSIFIED
![Project Screenshot](https://github.com/Slimsnapz/ision-Transformer--Image-Classification/blob/5c4523264ca1366eaf94d58cb56f853244673797/screenshots/Screenshot%202025-09-21%20072722.png)
*Training loss curve. Replace/add confusion matrix and demo screenshots before sharing.*


## TL;DR
Fine-tuned `google/vit-base-patch16-224-in21k` on the **Indian Food Images dataset (20 classes)** for dish classification.  
Includes reproducible training, evaluation, single-image inference, and a Streamlit demo for stakeholder inspection.  
Quick demo: `streamlit run src/app.py -- --model_dir models/indian_food_vit`



## Business problem & solution
**Problem:** Food-delivery platforms receive thousands of dish images uploaded by restaurants and customers. Manual labeling is slow, inconsistent, and doesn’t scale.  
**Solution:** This repo provides an end-to-end pipeline that fine-tunes a **Vision Transformer (ViT)** model to automatically classify dishes into 20 categories. The system improves catalog accuracy, enhances search/recommendations, and reduces operational costs by automating tagging.



## What’s inside (technical summary)
- **Dataset:** [rajistics/indian_food_images](https://huggingface.co/datasets/rajistics/indian_food_images) (20 dish categories).  
- **Model:** Fine-tuned `vit-base-patch16-224-in21k` with Hugging Face `Trainer` & `AutoModelForImageClassification`.  
- **Preprocessing:** RandomResizedCrop, normalization with processor means/std, conversion to pixel tensors.  
- **Training setup:**  
  - LR: `5e-5`, Batch: `16/4`, Epochs: `4`, GradAccum: `4`  
  - Metrics: Accuracy (primary), confusion matrix & per-class breakdowns.  
- **Evaluation:** Example results (replace with final numbers):  
  - Validation accuracy: **0.88**  
  - Test accuracy: **0.86**  
- **Deployment:** Lightweight **Streamlit demo** (`src/app.py`) for uploading images and getting predictions (top-k probabilities).



## Business impact
- **Efficiency:** Automates dish tagging at ingestion.  
- **Product quality:** Ensures consistent categories for better recommendations.  
- **Scalability:** Easily retrained as new data arrives.  



## Model hosting
The trained model (~321MB) is not tracked in Git. Download from Google Drive:  
🔗 [Drive folder link](https://drive.google.com/drive/folders/1uRqL_HsX_7a1NSVUug79N7Y6T7YeJchk?usp=sharing)  

After downloading to `models/indian_food_vit/`, load with:
```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
model = AutoModelForImageClassification.from_pretrained("models/indian_food_vit")
processor = AutoImageProcessor.from_pretrained("models/indian_food_vit")
