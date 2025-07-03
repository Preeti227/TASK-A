import os
import cv2
import numpy as np
from glob import glob
from src.filters import apply_all_filters

def get_label_from_path(path):
    return 0 if 'female' in path.lower() else 1

def load_and_filter_images_from_folder(folder_path):
    image_paths = glob(os.path.join(folder_path, '**', '*.*'), recursive=True)
    filtered_images, labels = [], []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filtered = apply_all_filters(img)
            filtered_images.append(filtered)
            labels.append(get_label_from_path(path))
    return np.array(filtered_images), np.array(labels)

def load_filtered_dataset(dataset_root):
    X_train, y_train = load_and_filter_images_from_folder(os.path.join(dataset_root, "train"))
    X_val, y_val = load_and_filter_images_from_folder(os.path.join(dataset_root, "val"))
    return X_train, y_train, X_val, y_val
