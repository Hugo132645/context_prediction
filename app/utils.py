import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from dataclasses import dataclass
from typing import Tuple, List
import math

@dataclass
class PatchConfig:
    patch_size: int = 32
    gap: int = 16
    jitter: int = 4
    max_image_size: Tuple[int, int] = (128, 128)
    color_drop: bool = True

DIRECTION_MAP = {
    0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
    3: ( 0, -1),             4: ( 0, 1),
    5: ( 1, -1), 6: ( 1, 0), 7: ( 1, 1)
}

transform = transforms.ToTensor()

def load_images_from_folder(folder: str, config: PatchConfig) -> List[Tuple[str, np.ndarray]]:
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, config.max_image_size)
            images.append((filename, img))
    return images

def apply_color_dropping(patch: np.ndarray) -> np.ndarray:
    keep_channel = np.random.randint(3)
    noise = np.random.normal(0, 1, patch.shape).astype(np.uint8)
    dropped = np.copy(patch)
    for c in range(3):
        if c != keep_channel:
            dropped[:, :, c] = noise[:, :, c]
    return dropped

def extract_random_patch_pair(img: np.ndarray, config: PatchConfig, apply_color_drop: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    h, w, _ = img.shape
    margin = config.patch_size + config.gap + config.jitter

    # Label → (dy, dx)
    DIRECTION_MAP = {
        0: ( 0,  1),  # Right
        1: (-1,  1),  # Top-Right
        2: (-1,  0),  # Top
        3: (-1, -1),  # Top-Left
        4: ( 0, -1),  # Left
        5: ( 1, -1),  # Bottom-Left
        6: ( 1,  0),  # Bottom
        7: ( 1,  1)   # Bottom-Right
    }

    shift = config.patch_size + config.gap

    for _ in range(20):  # try up to 20 times
        if h <= 2 * margin or w <= 2 * margin:
            raise ValueError("Image is too small to extract patches with the given config.")

        # Random anchor patch1
        x1 = np.random.randint(margin, w - margin)
        y1 = np.random.randint(margin, h - margin)

        dx_jit, dy_jit = np.random.randint(-config.jitter, config.jitter + 1, size=2)
        x1 += dx_jit
        y1 += dy_jit

        # Filter valid directions
        valid_directions = []
        for label, (dy, dx) in DIRECTION_MAP.items():
            x2 = x1 + dx * shift
            y2 = y1 + dy * shift
            if (0 <= x2 < w - config.patch_size and 0 <= y2 < h - config.patch_size):
                valid_directions.append((label, x2, y2))

        if not valid_directions:
            continue  # try a new p1

        # Randomly pick a valid direction
        direction, x2, y2 = valid_directions[np.random.randint(len(valid_directions))]

        patch1 = img[y1:y1 + config.patch_size, x1:x1 + config.patch_size]
        patch2 = img[y2:y2 + config.patch_size, x2:x2 + config.patch_size]

        if patch1.shape != (config.patch_size, config.patch_size, 3) or \
           patch2.shape != (config.patch_size, config.patch_size, 3):
            continue

        if apply_color_drop and config.color_drop:
            patch1 = apply_color_dropping(patch1)
            patch2 = apply_color_dropping(patch2)

        return patch1, patch2, direction

    raise ValueError("Failed to extract a valid patch pair after 20 attempts.")

def preprocess_patch_pair(p1: np.ndarray, p2: np.ndarray) -> torch.Tensor:
    t1 = transform(p1)
    t2 = transform(p2)
    return torch.cat([t1, t2], dim=0).unsqueeze(0)

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
