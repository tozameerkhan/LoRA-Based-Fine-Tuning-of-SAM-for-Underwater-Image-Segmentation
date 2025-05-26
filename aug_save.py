import os
import cv2
import numpy as np
from tqdm import tqdm
from augment import transform  # <-- import the transform we made

# Paths
original_images_dir = 'data/suim/train/images'  # <-- Change this
original_masks_dir = 'data/suim/train/masks'    # <-- Change this

augmented_images_dir = 'augN/images'  # <-- Change this
augmented_masks_dir = 'augN/masks'    # <-- Change this

os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_masks_dir, exist_ok=True)

# List images
image_filenames = [f for f in os.listdir(original_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
mask_filenames = [f for f in os.listdir(original_masks_dir) if f.endswith('.bmp')]

# Sort to align images/masks properly
image_filenames.sort()
mask_filenames.sort()

num_augmentations = 3
total_augmentations = 0
# Augmentation loop
for img_filename, mask_filename in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames)):

    # Get file extensions
    img_ext = os.path.splitext(img_filename)[1]
    mask_ext = os.path.splitext(mask_filename)[1]

    # Read
    img_path = os.path.join(original_images_dir, img_filename)
    mask_path = os.path.join(original_masks_dir, mask_filename)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)  # for masks, use grayscale

    for i in range(num_augmentations):
        #print(i)
        # Augment
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        # Generate save filenames
        img_basename = os.path.splitext(img_filename)[0]
        mask_basename = os.path.splitext(mask_filename)[0]

        # Save
        save_img_path = os.path.join(augmented_images_dir, f"aug_{i}_{img_basename}{img_ext}")
        save_mask_path = os.path.join(augmented_masks_dir, f"aug_{i}_{mask_basename}{mask_ext}")

        cv2.imwrite(save_img_path, aug_image)
        cv2.imwrite(save_mask_path, aug_mask)

        total_augmentations += 1


print(f"✅ Augmentation finished! Saved {total_augmentations} augmented images and masks.")
