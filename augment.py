import albumentations as A
import cv2

transform = A.Compose([
    # Spatial transformations
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=30,
        border_mode=cv2.BORDER_REFLECT,
        p=0.7
    ),
    
    # Underwater-specific augmentations
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
    
    # Simulating underwater conditions - fixed parameter names
    A.GaussNoise(var_limit=50, p=0.3),  # Fixed parameter
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Blur(blur_limit=7, p=0.2),  # Replacement for RandomFog
    
    # Careful distortions (fixed parameters)
    A.ElasticTransform(
        alpha=50,
        sigma=30,
        p=0.2
    ),
    
    # Consistent crop and resize - fixed to use size parameter
    A.RandomResizedCrop(
        size=(512, 512),  # Using size instead of height/width
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p=0.5
    ),
    
    # Final resize to ensure consistent dimensions
    A.Resize(height=512, width=512),
], is_check_shapes=False)