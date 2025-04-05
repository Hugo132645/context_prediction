import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from app.model import PatchNet
from app.utils import PatchConfig, load_images_from_folder, extract_random_patch_pair, preprocess_patch_pair, seed_everything

# Set seed and config
seed_everything(42)
config = PatchConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
images = load_images_from_folder("images", config)

# Define model
model = PatchNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):
    total_loss = 0
    for _ in range(100):  # number of iterations
        # Pick random image and patch pair
        _, img = images[torch.randint(len(images), (1,)).item()]
        try:
            p1, p2, label = extract_random_patch_pair(img, config)
        except ValueError:
            continue  # skip too-small images

        input_tensor = preprocess_patch_pair(p1, p2).to(device)
        label_tensor = torch.tensor([label], dtype=torch.long).to(device)

        # Forward pass
        output = model(input_tensor)
        loss = criterion(output, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
