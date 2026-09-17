# pip install albumentations opencv-python matplotlib pillow numpy

import os
import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Setup configuration
font_path = "tahoma.ttf"  # Windows: C:/Windows/Fonts/tahoma.ttf | Mac: /Library/Fonts/Tahoma.ttf
num_augmentations_per_char = 15

# Comprehensive character set: English + Thai Consonants, Vowels, Tone Marks, Numbers
english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
thai_chars = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"  # Consonants
    "ะาิีุูเแโใไำึื"  # Vowels
    "◌่◌้◌๊◌๋◌์◌็"  # Tone marks & signs
    "๐๑๒๓๔๕๖๗๘๙"  # Thai numbers
)
characters = english_chars + thai_chars

# Create output directories
os.makedirs("combined_alphabet_pngs", exist_ok=True)
os.makedirs("combined_augmented_images", exist_ok=True)

# 1. Generate base character PNGs
try:
  font = ImageFont.truetype(font_path, size=90)
  for char in characters:
    img = Image.new("RGBA", (150, 150), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((25, 15), char, fill=(0, 0, 0, 255), font=font)
    filename = f"combined_alphabet_pngs/{char.encode('utf-8').hex()}.png"
    img.save(filename)
  print("Base English & Thai PNGs generated successfully!")
except Exception as e:
  print(f"Error generating base images: {e}. Please check your font path.")

# 2. Define augmentation pipeline
transform = A.Compose(
    [
        A.ShiftScaleRotate(
            shift_limit=0.15, scale_limit=0.2, rotate_limit=30, p=0.9
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(5, 10),
            hole_width_range=(5, 10),
            fill=255,  # Fill dropped holes with white background color
            p=0.3,
        ),
    ]
)

# 3. Generate augmented versions with Outlier Filtering
total_generated = 0
total_outliers_rejected = 0

for char in characters:
  char_hex = char.encode("utf-8").hex()
  img_path = f"combined_alphabet_pngs/{char_hex}.png"
  
  if not os.path.exists(img_path):
    continue

  image = cv2.imread(img_path)
  if image is None:
    continue
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  valid_count = 0
  attempts = 0
  max_attempts = num_augmentations_per_char * 3  # Safeguard against infinite loops

  # Loop until we get the desired number of clean, non-outlier images
  while valid_count < num_augmentations_per_char and attempts < max_attempts:
    attempts += 1
    augmented = transform(image=image_rgb)
    augmented_image = augmented["image"]

    # --- OUTLIER DETECTION LOGIC ---
    # Convert to grayscale to evaluate pixel intensity distribution
    gray = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY)
    pixel_std = np.std(gray)

    # If standard deviation is too low, the image is practically blank 
    # (meaning the letter was entirely dropped out or pushed out of frame).
    if pixel_std < 5.0:
      total_outliers_rejected += 1
      continue  # Reject this outlier and try again

    # If it passes the filter, save it as a valid sample
    valid_count += 1
    aug_filename = f"combined_augmented_images/{char_hex}_aug_{valid_count}.png"
    cv2.imwrite(
        aug_filename, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    )
    total_generated += 1

print(
    f"\nSuccessfully generated {total_generated} clean augmented images!\n"
    f"🛡️ Filtered out and rejected {total_outliers_rejected} outlier/blank images."
)
