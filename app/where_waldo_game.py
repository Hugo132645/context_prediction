import streamlit as st
import torch
import numpy as np
import cv2
from model import PatchPositionNet
from utils import PatchConfig, load_images_from_folder, extract_random_patch_pair, preprocess_patch_pair

# Config
config = PatchConfig(patch_size=28, gap=8, jitter=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PatchPositionNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Load resized images
images = load_images_from_folder("images", config)
direction_names = [
    "Right", "Top-Right", "Top", "Top-Left",
    "Left", "Bottom-Left", "Bottom", "Bottom-Right"
]

# UI
st.title("Where's Waldo - Patch Direction Game")

# Select a random image
idx = np.random.randint(len(images))
filename, img = images[idx]
st.write(f"Image: {filename}")

# Extract patches
try:
    p1, p2, label = extract_random_patch_pair(img, config, apply_color_drop=False)
except ValueError:
    st.error("Image is too small for patch extraction.")
    st.stop()

# Add border to patches (but no text)
bordered_p1 = cv2.copyMakeBorder(p1, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 0, 0))  # Blue border
bordered_p2 = cv2.copyMakeBorder(p2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))  # Red border
combined = np.hstack([bordered_p1, bordered_p2])
st.image(combined, channels="BGR", width=256)

# Preprocess
input_tensor = preprocess_patch_pair(p1, p2).to(device)

# Predict
with torch.no_grad():
    logits = model(input_tensor)
    pred = torch.argmax(logits, dim=1).item()

# Buttons
st.subheader("Guess the direction of the red patch (from blue):")

col1, col2, col3 = st.columns(3)
guessed = st.session_state.get("guessed", False)

def guess(direction_idx):
    st.session_state.clicked = direction_idx
    st.session_state.guessed = True

with col1:
    if st.button("Top-Left"): guess(3)
    if st.button("Left"): guess(4)
    if st.button("Bottom-Left"): guess(5)

with col2:
    if st.button("Top"): guess(2)
    st.write(" ")
    if st.button("Bottom"): guess(6)

with col3:
    if st.button("Top-Right"): guess(1)
    if st.button("Right"): guess(0)
    if st.button("Bottom-Right"): guess(7)

# Feedback
if st.session_state.get("guessed", False):
    user_dir = direction_names[st.session_state.clicked]
    correct_dir = direction_names[label]
    model_dir = direction_names[pred]

    st.markdown("---")
    if st.session_state.clicked == label:
        st.success(f"✅ Correct! You picked: **{user_dir}**")
    else:
        st.error(f"❌ Wrong. You picked: **{user_dir}** — Correct: **{correct_dir}**")

    st.info(f"🤖 Model predicted: **{model_dir}**")

    if st.button("Play Again"):
        st.session_state.guessed = False
        st.rerun()
