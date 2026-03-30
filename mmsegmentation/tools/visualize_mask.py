" Script for visualising masks on images to check segmentation by mmsegmentation trained models"
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Define paths to image and mask
img_path = "/home/marloes.stemerdink@mydre.org/Documents/DCM_test/png/1.2.392.200036.9116.6.22.11522156.9942.20250502070844414.2.73.png"
mask_path = "/home/marloes.stemerdink@mydre.org/Documents/DCM_test/pred/1.2.392.200036.9116.6.22.11522156.9942.20250502070844414.2.73.png"

# Load image and mask
img = np.array(Image.open(img_path).convert("RGB"))
mask = np.array(Image.open(mask_path))  # 0 = background, 1 = foreground

# Create a red overlay where mask == 1
overlay = img.copy()
overlay[mask == 1] = [255, 0, 0]

alpha = 0.4  # transparency
blended = (alpha * overlay + (1 - alpha) * img).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.title("Image"); plt.imshow(img); plt.axis("off")
plt.subplot(1, 3, 2); plt.title("Mask"); plt.imshow(mask, cmap="gray"); plt.axis("off")
plt.subplot(1, 3, 3); plt.title("Overlay"); plt.imshow(blended); plt.axis("off")
plt.show()

