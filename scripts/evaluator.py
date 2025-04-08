import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from app.model import PatchPositionNet
from app.utils import PatchConfig, load_images_from_folder, extract_random_patch_pair, preprocess_patch_pair

def evaluate_model(model, img, config, test_pairs=1000):  
    # Load resized images
    images = load_images_from_folder("images", config)
    y_preds = []
    y_true = []
    direction_names = [
        "Right", "Top-Right", "Top", "Top-Left",
        "Left", "Bottom-Left", "Bottom", "Bottom-Right"
    ]

    for _ in range(test_pairs):
        # Select a random image
        idx = np.random.randint(len(images))
        filename, img = images[idx]
        
        try:
            p1, p2, label = extract_random_patch_pair(img, config, apply_color_drop=False)
        except ValueError:
            raise Exception("Image is too small for patch extraction.")
        
        # Preprocess
        input_tensor = preprocess_patch_pair(p1, p2).to(device)

        # Predict
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()
        
        # Store predictions and labels
        y_preds.append(pred)
        y_true.append(label)
    
    # Print classification report
    print(classification_report(y_true, y_preds, target_names=direction_names))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_preds)
    
    # Show confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=direction_names)
    disp.plot()
    plt.show()
    
    
if __name__ == "__main__":
    # Config
    config = PatchConfig(patch_size=28, gap=8, jitter=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PatchPositionNet().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    
    evaluate_model(model, "images", config)
    
        
        
    