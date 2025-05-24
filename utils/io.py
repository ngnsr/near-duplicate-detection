import os
import zipfile
from typing import List
import streamlit as st

def save_uploaded_images(uploaded_files, temp_dir: str) -> List[str]:
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)
    return image_paths

def extract_zip_and_get_images(uploaded_zip, temp_dir: str) -> List[str]:
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    image_paths = [
        os.path.join(temp_dir, f)
        for f in os.listdir(temp_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return image_paths

def clear_temp_directory(temp_dir: str):
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
