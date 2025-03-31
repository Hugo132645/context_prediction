import cv2
import numpy as np
import os
from torchvision import transforms
import torch

PATCH_SIZE = 64
GAP = 32  # gap between patches

DIRECTION_MAP = {
    0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
    3: (0, -1),             4: (0, 1),
    5: (1, -1),  6: (1, 0), 7: (1, 1)
}

def load_images_from_folder(folder, max_size=(300, 300)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, max_size)
            images.append((filename, img))
    return images

def extract_random_patch_pair(img):
    h, w, _ = img.shape
    margin = PATCH_SIZE + GAP
    x = np.random.randint(margin, w - margin)
    y = np.random.randint(margin, h - margin)

    # choose a direction
    direction = np.random.randint(8)
    dy, dx = DIRECTION_MAP[direction]

    patch1 = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
    x2 = x + dx * (PATCH_SIZE + GAP)
    y2 = y + dy * (PATCH_SIZE + GAP)
    patch2 = img[y2:y2+PATCH_SIZE, x2:x2+PATCH_SIZE]

    return patch1, patch2, direction

def preprocess_patch_pair(p1, p2):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    p1 = transform(p1)
    p2 = transform(p2)
    return torch.cat([p1, p2], dim=0).unsqueeze(0)
