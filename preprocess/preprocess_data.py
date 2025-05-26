import os
import argparse
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, default='data/suim/train1', help='Path to SUIM data')
parser.add_argument('--dst_path', type=str, default='data/suim_npz1', help='Path to save npz files')
parser.add_argument('--image_size', type=int, default=512, help='Resize images/masks to this size')
parser.add_argument('--normalize', action='store_true', default=True, help='Normalize image pixel values to [0, 1]')
args = parser.parse_args()

os.makedirs(args.dst_path, exist_ok=True)

image_paths = sorted(glob(os.path.join(args.src_path, 'images', '*.jpg')))
mask_paths = sorted(glob(os.path.join(args.src_path, 'masks', '*.bmp')))

print(f"Found {len(image_paths)} image-mask pairs")

for img_path, msk_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Processing"):
    img = Image.open(img_path).convert("RGB").resize((args.image_size, args.image_size))
    msk = Image.open(msk_path).convert("RGB").resize((args.image_size, args.image_size), Image.NEAREST)

    img_np = np.array(img).astype(np.float32)
    msk_np = np.array(msk).astype(np.float32)

    if args.normalize:
        img_np /= 255.0

    save_name = os.path.basename(img_path).replace(".jpg", ".npz")
    save_path = os.path.join(args.dst_path, save_name)

    np.savez_compressed(save_path, image=img_np, label=msk_np)

print(f"✅ Preprocessing complete. Saved {len(image_paths)} npz files to: {args.dst_path}")
