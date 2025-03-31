# Where's Waldo: Meme Patch Game

A self-supervised learning project that trains a neural network to guess which image patch goes where — using memes instead of Waldo books.

Built for an interactive presentation and laugh-powered machine learning.

---

## What It Does

- Loads real memes and cuts them into shuffled patches.
- Trains a lightweight neural network to predict the position of a patch based on its neighbors.
- Lets you play an interactive "Where's Waldo?" game powered by this model in Streamlit.

---

## How It Works

The idea comes from unsupervised context prediction (like [Doersch et al. 2015](https://arxiv.org/abs/1505.05192)), but applied to meme images for fun.

The model learns to identify the position of a patch (e.g., “is this top-left or bottom-right?”) by comparing it to other patches nearby.

---

## Requirements

- Python 3.10
- PyTorch ≥ 2.2
- NumPy 1.x
- OpenCV
- Pillow
- Streamlit

I had some issues with Python 3.12 and Numpy 2.x since I am using an old Intel-based Mac, but you could adjust the requirements according to your needs.

---

## How to Run It

### 1. Train the PatchNet
```bash
python train_patch_net.py
