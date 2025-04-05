import os
import cv2
import numpy as np
import requests
from io import BytesIO

# Create image directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# List of meme URLs
meme_urls = [
    "https://i.imgflip.com/1bij.jpg",
    "https://i.imgflip.com/30b1gx.jpg",
    "https://i.imgflip.com/4t0m5.jpg",
    "https://i.imgflip.com/26am.jpg",
    "https://i.imgflip.com/1ur9b0.jpg",
    "https://i.imgflip.com/2fm6x.jpg",
    "https://i.imgflip.com/3si4.jpg",
    "https://i.imgflip.com/1otk96.jpg"
]

# Target resize dimensions
RESIZE_DIM = (128, 128)

for idx, url in enumerate(meme_urls, start=1):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Failed to decode image {idx}")
            continue

        # Resize
        resized = cv2.resize(image, RESIZE_DIM)

        # Save to images folder
        filename = f"images/meme_{idx}.png"
        cv2.imwrite(filename, resized)
        print(f"Downloaded and resized meme {idx} → {filename}")
    except Exception as e:
        print(f"Error downloading meme {idx} from {url}: {e}")
