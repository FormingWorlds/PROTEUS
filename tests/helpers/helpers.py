from __future__ import annotations

from PIL import Image
from pathlib import Path

PROTEUS_ROOT = Path(__file__).parents[2]

def resize_to_match(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Resize img2 to match img1's size if they don't match
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    return img1, img2
