#solely on generating the alphabet pngs for the augmentation process

import os
from PIL import Image, ImageDraw, ImageFont

# Create output directory
os.makedirs("alphabet_pngs", exist_ok=True)

# Load a TrueType font (update path to a font on your system)
# Windows: C:/Windows/Fonts/arial.ttf | Mac: /Library/Fonts/Arial.ttf | Linux: /usr/share/fonts/...
font_path = "arial.ttf"
font = ImageFont.truetype(font_path, size=120)

english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
thai_chars = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"  # Consonants
    "ะาิีุูเแโใไำึื"  # Vowels
    "◌่◌้◌๊◌๋◌์◌็"  # Tone marks & signs (with dotted circle for clear rendering)
    "๐๑๒๓๔๕๖๗๘๙"  # Thai numbers
)
characters = english_chars + thai_chars

for char in characters:
  # Create a blank image with a transparent background (RGBA)
  img = Image.new("RGBA", (150, 150), (255, 255, 255, 0))
  draw = ImageDraw.Draw(img)

  # Draw the character
  draw.text((35, 10), char, fill=(0, 0, 0, 255), font=font)

  # Save as PNG
  filename = f"alphabet_pngs/{ord(char)}_{char}.png"
  img.save(filename)

print("Alphabet PNGs generated successfully!")