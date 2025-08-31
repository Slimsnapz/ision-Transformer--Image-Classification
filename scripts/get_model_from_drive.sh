#!/usr/bin/env bash
# scripts/get_model_from_drive.sh
# Usage: ./scripts/get_model_from_drive.sh <gdrive-file-id> <out_zip>
# Example: ./scripts/get_model_from_drive.sh 1a2b3C4d5Ef6 model.zip

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <gdrive-file-id> <out_zip>"
  exit 1
fi

FILE_ID="$1"
OUT_ZIP="$2"

# Ensure gdown is installed
pip install --upgrade gdown

echo "Downloading file id $FILE_ID to $OUT_ZIP ..."
gdown --id "$FILE_ID" -O "$OUT_ZIP"

if [ $? -ne 0 ]; then
  echo "Download failed. Check the file id and that the file is shared publicly."
  exit 2
fi

echo "Unzipping to models/indian_food_vit ..."
mkdir -p models/indian_food_vit
unzip -o "$OUT_ZIP" -d models/indian_food_vit

echo "Done. Model files extracted to models/indian_food_vit/"
