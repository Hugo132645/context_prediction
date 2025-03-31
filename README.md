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

The idea comes from unsupervised context prediction (like [Doersch et al. 2015]([https://arxiv.org/abs/1505.05192](https://ieeexplore.ieee.org/document/7410524))), but applied to meme images for fun.

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

## Team Members

- Hugo Arsenio – Training system implementation
- Dan Angel – Utils and data pipeline
- Niko – Evaluation of the model

---

## Project Structure

```
context_prediction/
├── app/
│   ├── where_waldo_game.py     # Streamlit game app
│   ├── model.py                # PatchNet model definition
│   └── utils.py                # Patch slicing and utilities
│
├── scripts/
│   ├── download_memes.py       # Downloads meme images
│   ├── train_patch_net.py      # Training script
│   └── evaluator.py            # Model evaluation script
│
├── images/                     # Downloaded meme images (gitignored)
├── model.pth                   # Trained model weights (gitignored)
├── requirements.txt            # Python dependency list
├── .gitignore                  # Files/folders to exclude from Git
└── README.md                   # Project overview and instructions
```

## How to Run the Project

> All steps assume you're working in a Python 3.10+ environment with PyTorch 2.2+, NumPy 1.x, etc.

---

### Install Dependencies

We recommend using a Conda environment:

```bash
conda create -n waldo_env python=3.10 -y
conda activate waldo_env

pip install -r requirements.txt
```

### Download Meme Images

You can use your own images or find some cool ones to use. Just put the URLs in the ```download_memes.py``` file and it will download them all in the images folder for the model.

```bash
python download_memes.py
```

### Train the PatchNet Model

Train a model to guess where an image patch comes from:

```bash
python train_patch_net.py
```

After training, a ```model.pth``` file will be saved. This file is used by the game to make predictions.

### Play the Game!

Once trained, launch the Streamlit game:

```bash
streamlit run where_waldo_game.py
```

You’ll be shown a meme with one shuffled patch, and you’ll have to guess where it belongs.
The AI will try to predict that too — who’s better, you or the model?

### Notes

The model is CPU-friendly and trains in a few minutes on small datasets.

You can replace the meme images in ```images/``` with your own for custom training and gameplay.

The game UI is built with Streamlit and is designed to be presentation-ready for demo purposes.
