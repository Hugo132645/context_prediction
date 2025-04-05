import streamlit as st
import cv2
import numpy as np
import torch
from app.model import PatchNet
from app.utils import PatchConfig, load_images_from_folder, extract_random_patch_pair, DIRECTION_MAP

# Setup
st.title("Where's Waldo: Patch Prediction Game")
config = PatchConfig()
device = torch.device("cpu")

# Load model
model = PatchNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Load image
images = load_images_from_folder("images", config)
filename, img = images[np.random.randint(len(images))]

# Extract patch pair
patch1, patch2, label = extract_random_patch_pair(img, config)

# Preprocess
input_tensor = torch.cat([torch.tensor(patch1).permute(2,0,1)/255.0,
                          torch.tensor(patch2).permute(2,0,1)/255.0], dim=0).unsqueeze(0).float()
output = model(input_tensor)
predicted_direction = torch.argmax(output).item()

# Show patch
st.image(cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB), caption="Mystery Patch", width=200)

# Let user guess
guess = st.radio("Where is this patch from?", list(DIRECTION_MAP.keys()))

# Button to submit guess
if st.button("Submit Guess"):
    if guess == label:
        st.success("Correct!")
    else:
        st.error(f"Incorrect. Correct answer: {label}")

    if guess == predicted_direction:
        st.info("Model agrees with you!")
    else:
        st.warning(f"Model guessed: {predicted_direction}")

# Show full image with outline
def draw_outline(img, center, color):
    x, y = center
    return cv2.rectangle(img.copy(), (x, y), (x + config.patch_size, y + config.patch_size), color, 2)

# Compute true location
margin = config.patch_size + config.gap
center_x = img.shape[1] // 2
center_y = img.shape[0] // 2
dx, dy = DIRECTION_MAP[label]
tx = center_x + dx * (config.patch_size + config.gap)
ty = center_y + dy * (config.patch_size + config.gap)

outlined = draw_outline(img, (tx, ty), (255, 0, 0))  # Red for true patch
st.image(cv2.cvtColor(outlined, cv2.COLOR_BGR2RGB), caption="Meme with Correct Patch Location")
