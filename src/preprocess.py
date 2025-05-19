#!/usr/bin/env python3
"""
preprocess.py

Split raw PNG images into train, validate, and test; compute global RGB 
statistics, and save them for downstream use.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

import os
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.format import *


# ──────────────────────────────────────────────────────────────────────────────
# Erase already existing images
# ──────────────────────────────────────────────────────────────────────────────

def clear_folder(path: Path) -> None:
    """
    Clear folder and leave it empty.
    """
    
    message = f"{"Clearing folder":20.20}"
    for entry in tqdm(os.listdir(path), desc=message):
        entry_path = os.path.join(path, entry)
        
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.remove(entry_path)
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
        except Exception as e:
            print(f"{BOLD_RED}Failed to delete {entry_path}: {e}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Image Loading & Statistics
# ──────────────────────────────────────────────────────────────────────────────

def load_images(path: Path) -> np.ndarray:
    """
    Load images and stack them.
    """
    
    filenames = sorted(path.glob("*.png"))

    images = []

    message = f"{"Loading images":20.20}"
    for filename in tqdm(filenames, desc=message):
        image = Image.open(filename).convert("RGB")
        image_rgb_array = np.array(image)
        images.append(image_rgb_array)
        
    images = np.stack(images, axis=0)
    return images


def compute_image_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes per-channel mean and standard deviation in [0, 1].
    """
    
    images = images.astype(np.float32) / 255.0
    
    mean = images[:, :, :, :].mean(axis=(0, 1, 2))
    std = images[:, :, :, :].std(axis=(0, 1, 2))

    return mean, std


# ──────────────────────────────────────────────────────────────────────────────
# Splitting Image Functions
# ──────────────────────────────────────────────────────────────────────────────

def split_images(negative_images: np.ndarray, positive_images: np.ndarray,
    train_fraction: float, validate_fraction: float, test_fraction: float) -> \
        None:
    """
    Split each class subfolder into train/validate/test and save as PNGs.
    """
    
    assert(abs(train_fraction + validate_fraction + test_fraction - 1.0) < 1e-6)
    
    classes = {
        "positive": "positive",
        "negative": "negative"
    }
    
    raw_dir       = Path("data/raw")
    processed_dir = Path("data/processed")
    
    message = f"{"Splitting images":20.20}"
    for raw_subdir, processed_subdir in tqdm(classes.items(), \
                                             desc=message):
        files = sorted((raw_dir/raw_subdir).glob("*.png"))
        
        # First split (train and "other")
        train_files, other_files = train_test_split(
            files,
            train_size = train_fraction,
            random_state = 42
        )
        
        rest_of_fraction = validate_fraction + test_fraction
            
        # Second split (validate and test)
        validate_files, test_files = train_test_split(
            other_files,
            train_size = validate_fraction / rest_of_fraction,
            random_state = 42
        )
        
        # Give names to each file list
        splits = {
            "train":    train_files,
            "validate": validate_files,
            "test":     test_files,
        }
        
        # Place the images in their designated folders
        for split, file_list in splits.items():
            output_dir = processed_dir / split / processed_subdir
            output_dir.mkdir(
                parents = True,
                exist_ok = True
            )
            
            for file in file_list:
                image = Image.open(file).convert("RGB")
                image.save(output_dir / file.name)
                
# ──────────────────────────────────────────────────────────────────────────────
# Preprocess Function
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(split_fractions=[0.70, 0.15, 0.15]) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete function that performs all the preprocessing.
    """
    
    message = f"{DIM}%(asctime)s{RESET}\t\t{BOLD_YELLOW}%(message)s{RESET}"
    dashes = "-" * TERMINALSIZE
    logging.basicConfig(
        level = logging.INFO,
        format = f"\n\n{message}\n{dashes}"
    )

    logging.info("Emptying processed subfolders...")
    clear_folder("data/processed/train/negative")
    clear_folder("data/processed/train/positive")
    clear_folder("data/processed/validate/negative")
    clear_folder("data/processed/validate/positive")
    clear_folder("data/processed/test/negative")
    clear_folder("data/processed/test/positive")

    logging.info("Loading raw images into variables...")
    negative_images = load_images(Path("data/raw/negative"))
    positive_images = load_images(Path("data/raw/positive"))

    logging.info("Computing images' mean and standard deviation...")
    images = np.concatenate((negative_images, positive_images), axis=0)
    mean, std = compute_image_stats(images)

    logging.info("Splitting images into train/validate/test subfolders in processed...")
    train_fraction, validate_fraction, test_fraction = split_fractions
    split_images(
        negative_images, positive_images,
        train_fraction, validate_fraction, test_fraction
    )
    
    logging.info(f"{BOLD_GREEN}Preprocessing complete!{RESET}")
    
    return mean, std