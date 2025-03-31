import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from utils import DIRECTION_MAP, PATCH_SIZE, GAP
from model import PatchPositionNet

class PatchDataset(Dataset):
    def __init__(self, img_dir, num_pairs=10000):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.num_pairs = num_pairs

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        img_path = np.random.choice(self.img_paths)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 300))

        h, w, _ = img.shape
        margin = PATCH_SIZE + GAP
        x = np.random.randint(margin, w - margin)
        y = np.random.randint(margin, h - margin)
        direction = np.random.randint(8)
        dy, dx = DIRECTION_MAP[direction]

        patch1 = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        x2 = x + dx * (PATCH_SIZE + GAP)
        y2 = y + dy * (PATCH_SIZE + GAP)
        patch2 = img[y2:y2+PATCH_SIZE, x2:x2+PATCH_SIZE]

        if patch1.shape != (PATCH_SIZE, PATCH_SIZE, 3) or patch2.shape != (PATCH_SIZE, PATCH_SIZE, 3):
            return self.__getitem__(idx)  #skip broken samples

        p1 = torch.from_numpy(patch1.transpose(2, 0, 1)).float() / 255.0
        p2 = torch.from_numpy(patch2.transpose(2, 0, 1)).float() / 255.0
        pair = torch.cat([p1, p2], dim=0)
        return pair, direction

def train():
    model = PatchPositionNet()
    dataset = PatchDataset("images", num_pairs=10000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
