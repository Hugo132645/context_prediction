import streamlit as st
import torch
import numpy as np
import cv2
from model import PatchPositionNet
from utils import PatchConfig, load_images_from_folder, extract_random_patch_pair, preprocess_patch_pair, DIRECTION_MAP

# Config with smaller patch and gap to work with small images
config = PatchConfig(patch_size=28, gap=8, jitter=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PatchPositionNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Load resized images
images = load_images_from_folder("images", config)
direction_names = [
    "Top-Left", "Top", "Top-Right",
    "Left",           "Right",
    "Bottom-Left", "Bottom", "Bottom-Right"
]

# UI
st.title("Where's Waldo - Patch Direction Game")

# Select a random image
idx = np.random.randint(len(images))
filename, img = images[idx]
st.write(f"Selected image: {filename}, shape: {img.shape}")

# Extract patches
try:
    p1, p2, label = extract_random_patch_pair(img, config, apply_color_drop=False)
except ValueError:
    st.error("Image is too small for patch extraction.")
    st.stop()

# Preprocess
input_tensor = preprocess_patch_pair(p1, p2).to(device)

# Prediction
with torch.no_grad():
    logits = model(input_tensor)
    pred = torch.argmax(logits, dim=1).item()

# Display patches side by side
combined = np.hstack([
    cv2.copyMakeBorder(p1, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 0, 0)),  # Blue border
    cv2.copyMakeBorder(p2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))   # Red border
])
st.image(combined, caption=f"Image: {filename}", channels="BGR", width=256)

# Show buttons
st.subheader("Guess the relative direction of the second patch (red) from the first (blue):")

col1, col2, col3 = st.columns(3)
guessed = st.session_state.get("guessed", False)
correct = st.session_state.get("correct", None)

def guess(direction_idx):
    st.session_state.guessed = True
    st.session_state.correct = (direction_idx == label)

with col1:
    if st.button("Top-Left"): guess(0)
    if st.button("Left"): guess(3)
    if st.button("Bottom-Left"): guess(5)

with col2:
    if st.button("Top"): guess(1)
    st.write(" ")  # Spacer
    if st.button("Bottom"): guess(6)

with col3:
    if st.button("Top-Right"): guess(2)
    if st.button("Right"): guess(4)
    if st.button("Bottom-Right"): guess(7)

# Feedback
if st.session_state.get("guessed", False):
    st.markdown("---")
    if st.session_state["correct"]:
        st.success("✅ Correct!")
    else:
        correct_dir = direction_names[label]
        predicted_dir = direction_names[pred]
        st.error(f"❌ Wrong. Correct: **{correct_dir}** — Model predicted: **{predicted_dir}**")

    if st.button("Play Again"):
        st.session_state.guessed = False
        st.rerun()
