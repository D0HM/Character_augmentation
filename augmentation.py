# pip install albumentations opencv-python matplotlib pillow numpy

import os
import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Setup configuration
font_path = "tahoma.ttf"  # Windows: C:/Windows/Fonts/tahoma.ttf | Mac: /Library/Fonts/Tahoma.ttf
num_augmentations_per_char = 20

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
os.makedirs("max_distorted_alphabet_pngs", exist_ok=True)
os.makedirs("max_distorted_output", exist_ok=True)

# 1. Generate base character PNGs with a Drop-Shadow effect
try:
  font = ImageFont.truetype(font_path, size=90)
  for char in characters:
    img = Image.new("RGBA", (150, 150), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Offset shadow layer
    draw.text((28, 18), char, fill=(160, 160, 160, 255), font=font)
    # Main black character layer
    draw.text((25, 15), char, fill=(0, 0, 0, 255), font=font)
    
    rgb_img = img.convert("RGB")
    filename = f"max_distorted_alphabet_pngs/{char.encode('utf-8').hex()}.png"
    rgb_img.save(filename)
    
  print("Base shadowed character PNGs generated successfully!")
except Exception as e:
  print(f"Error generating base images: {e}. Please check your font path.")

# 2. Maximum Distortion Pipeline ("Extreme Warp & Tear")
transform = A.Compose(
    [
        # --- EXTREME SPATIAL & PERSPECTIVE WARPS ---
        A.ShiftScaleRotate(
            shift_limit=0.25,
            scale_limit=0.3,
            rotate_limit=60,  # Up to 60-degree rotations
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
            p=0.95,
        ),
        A.Perspective(scale=(0.1, 0.25), pad_val=255, p=0.7),  # Heavy 3D tilt
        A.ElasticTransform(
            alpha=2.0, sigma=40, alpha_affine=40,  # Heavy rubber-like bending
            border_mode=cv2.BORDER_CONSTANT, value=255, p=0.6
        ),
        A.GridDistortion(
            num_steps=5, distort_limit=0.6,  # Strong grid-based wave warping
            border_mode=cv2.BORDER_CONSTANT, value=255, p=0.6
        ),
        A.OpticalDistortion(
            distort_limit=0.6, shift_limit=0.2,  # Fisheye lens-style distortion
            border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5
        ),

        # --- LIGHTING, CONTRAST & INTENSITY ---
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
        A.RandomGamma(gamma_limit=(60, 140), p=0.6),

        # --- BLUR & SHARPNESS ---
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(5, 9), p=1.0),  # Heavy motion blur streaks
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.6,
        ),

        # --- NOISE & ARTIFACTS ---
        A.GaussNoise(var_limit=(20.0, 70.0), p=0.5),  # Heavy pixel grain
        A.ImageCompression(quality_range=(40, 85), p=0.5),  # Heavy JPEG artifacting

        # --- SEVERE INK DROPOUTS (Faded/Bitten Strokes) ---
        A.CoarseDropout(
            num_holes_range=(2, 5),
            hole_height_range=(5, 12),
            hole_width_range=(5, 12),
            fill=255,
            p=0.6,
        ),
        A.GridDropout(
            ratio=0.15, unit_size_min=4, unit_size_max=10,
            shift_x=0, shift_y=0, fill_value=255, p=0.4
        ),
    ]
)

# 3. Execution Loop with Relaxed Outlier Safeguards (to accommodate extreme warping)
total_generated = 0
total_rejected = 0

for char in characters:
  char_hex = char.encode("utf-8").hex()
  img_path = f"max_distorted_alphabet_pngs/{char_hex}.png"
  
  if not os.path.exists(img_path):
    continue

  image = cv2.imread(img_path)
  if image is None:
    continue
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  valid_count = 0
  attempts = 0
  max_attempts = num_augmentations_per_char * 8  # Higher attempt cap due to aggressive filtering

  while valid_count < num_augmentations_per_char and attempts < max_attempts:
    attempts += 1
    augmented = transform(image=image_rgb)
    augmented_image = augmented["image"]

    # --- OUTLIER DETECTION (Adjusted for Extreme Distortion) ---
    gray = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Ink density check (wider bounds because warping stretches or shrinks characters)
    black_pixel_count = int(np.sum(thresh == 255))
    total_pixels = int(thresh.shape[0] * thresh.shape[1])
    ink_density = float(black_pixel_count) / float(total_pixels)

    if ink_density < 0.01 or ink_density > 0.65:
      total_rejected += 1
      continue

    # Component fragment check (allows slightly more fragmentation due to heavy elastic tearing)
    num_labels, _ = cv2.connectedComponents(thresh)
    if num_labels > 9:
      total_rejected += 1
      continue

    # Save valid highly-distorted sample
    valid_count += 1
    out_filename = f"max_distorted_output/{char_hex}_aug_{valid_count}.png"
    cv2.imwrite(
        out_filename, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    )
    total_generated += 1

print(
    f"\n💥 Maximum distortion pipeline complete!\n"
    f"📁 Successfully saved {total_generated} heavily warped images to 'max_distorted_output/'.\n"
    f"🛡️ Filtered out and discarded {total_rejected} completely destroyed/blank outliers."
)
