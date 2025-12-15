import os
import random
import shutil
from glob import glob

src_dir = "/home/megrad/Documents/Github/lejepa/ds/val/rgb_images/rgb"
out_dir = "/home/megrad/Documents/Github/lejepa/ds/test_images/rgb"

os.makedirs(out_dir, exist_ok=True)

image_paths = []

for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".tif", ".tiff"):
    image_paths.extend(glob(os.path.join(src_dir, f"*{ext}")))

random.shuffle(image_paths)

subset_size = int(len(image_paths)) // 3
subset = random.sample(image_paths, subset_size)

for img in subset:
    shutil.copy2(img, out_dir)
