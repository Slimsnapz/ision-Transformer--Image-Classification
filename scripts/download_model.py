#!/usr/bin/env python3
# scripts/download_model.py
# Simple helper to download a Google Drive file using gdown and unzip it.
# Usage: python scripts/download_model.py --file-id <FILE_ID> --out indian_food_vit.zip

import argparse
import subprocess
import os
import sys

def install_gdown():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])

def download(file_id, out):
    install_gdown()
    cmd = ["gdown", "--id", file_id, "-O", out]
    subprocess.check_call(cmd)

def unzip_file(zip_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    cmd = ["unzip", "-o", zip_path, "-d", dest_dir]
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True, help="Google Drive file id for the zipped model")
    parser.add_argument("--out", default="indian_food_vit.zip", help="Output zip name")
    parser.add_argument("--dest", default="models/indian_food_vit", help="Destination directory to extract files")
    args = parser.parse_args()

    download(args.file_id, args.out)
    unzip_file(args.out, args.dest)
    print("Model extracted to", args.dest)

if __name__ == "__main__":
    main()
