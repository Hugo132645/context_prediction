import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import PatchPositionNet
from utils import PatchConfig, load_images_from_folder, extract_random_patch_pair, preprocess_patch_pair, seed_everything

# Set up config and seed
config = PatchConfig(
    patch_size=64,
    gap=32,
    jitter=5,
    max_image_size=(300, 300),
    color_drop=True
)
seed_everything(42)

# Dataset Class
class PatchPairDataset(Dataset):
    def __init__(self, image_folder, config, num_pairs=10000):
        self.config = config
        self.images = load_images_from_folder(image_folder, config)
        self.num_pairs = num_pairs
        if not self.images:
            raise RuntimeError(f"No valid images found in {image_folder}")

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        for _ in range(10):  # retry up to 10 times
            try:
                _, img = self.images[torch.randint(len(self.images), (1,)).item()]
                p1, p2, direction = extract_random_patch_pair(img, self.config)
                input_tensor = preprocess_patch_pair(p1, p2).squeeze(0)
                return input_tensor, direction
            except ValueError:
                continue
        raise RuntimeError("Failed to extract a valid patch pair after 10 attempts.")

# Training Function
def train():
    dataset = PatchPairDataset("images", config, num_pairs=5000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = PatchPositionNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
