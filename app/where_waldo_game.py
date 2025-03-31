import streamlit as st
import torch
import numpy as np
import random
import cv2
from PIL import Image
from model import PatchPositionNet
from utils import load_images_from_folder, extract_random_patch_pair, preprocess_patch_pair

# --- Setup ---
st.set_page_config(page_title="Where's the Patch?", layout="centered")
st.title("Where's the Patch?")
st.caption("Can you outsmart a neural net? Guess where the red patch came from!")

DIRECTIONS = [
    "Top-Left", "Top", "Top-Right",
    "Left",         "Right",
    "Bottom-Left", "Bottom", "Bottom-Right"
]

# --- Load model ---
model = PatchPositionNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# --- Load images ---
images = load_images_from_folder("images")
filename, image = random.choice(images)

# --- Get patch pair and options ---
patch1, patch2, correct_dir = extract_random_patch_pair(image)
patch_pairs = []
for i in range(8):
    # generate all possible second patches
    fake_patch1, fake_patch2, _ = extract_random_patch_pair(image)
    patch_pairs.append((patch1, fake_patch2))
patch_pairs[correct_dir] = (patch1, patch2)

# --- Show anchor patch (blue) ---
st.subheader("🔵 Reference Patch")
st.image(patch1, width=128)

# --- Show 8 candidate patches as buttons ---
st.subheader("🔴 Where's the red patch?")
cols = st.columns(4)
user_guess = None

for i in range(8):
    with cols[i % 4]:
        if st.button(DIRECTIONS[i]):
            user_guess = i

# --- Run model prediction ---
guesses = []
for p1, p2 in patch_pairs:
    input_tensor = preprocess_patch_pair(p1, p2)
    with torch.no_grad():
        output = model(input_tensor)
    prob = torch.softmax(output, dim=1)
    guesses.append(prob[0, i].item())

model_prediction = torch.argmax(torch.tensor(guesses)).item()

# --- Show results ---
if user_guess is not None:
    st.markdown("### Results")
    st.write(f" **Correct answer:** {DIRECTIONS[correct_dir]}")
    st.write(f" **Model guessed:** {DIRECTIONS[model_prediction]}")
    st.write(f" **You guessed:** {DIRECTIONS[user_guess]}")
    
    if user_guess == correct_dir:
        st.success("You nailed it!")
    else:
        st.warning("Not quite! But hey, even the model might be wrong")

    if model_prediction == correct_dir:
        st.info("The model got it right too!")
    else:
        st.info("The model was confused too. Great minds think alike?")
