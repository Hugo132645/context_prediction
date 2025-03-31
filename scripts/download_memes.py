import os
import requests
from PIL import Image
from io import BytesIO

MEME_URLS = [
    "https://i.imgflip.com/1bij.jpg",       
    "https://i.imgflip.com/30b1gx.jpg",     
    "https://i.imgflip.com/4t0m5.jpg",      
    "https://i.imgflip.com/26am.jpg",       
    "https://i.imgflip.com/1ur9b0.jpg",     
    "https://i.imgflip.com/2fm6x.jpg",      
    "https://i.imgflip.com/3si4.jpg",        
    "https://i.imgflip.com/1otk96.jpg",      
]

SAVE_DIR = "images"

def download_memes():
    os.makedirs(SAVE_DIR, exist_ok=True)
    for idx, url in enumerate(MEME_URLS):
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            path = os.path.join(SAVE_DIR, f"meme_{idx+1}.png")
            img.save(path)
            print(f"Downloaded meme {idx+1} → {path}")
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

if __name__ == "__main__":
    download_memes()
