import cv2
import numpy as np
import sys

# 1. Load Image
img_path = 'input.jpg'
# Read as Grayscale (0 flag). 2D Convolution is easier to learn on 1 channel first.
img = cv2.imread(img_path, 0) 

if img is None:
    print("Error: Could not find input.jpg")
    sys.exit()

height, width = img.shape
print(f"Image Loaded: {width}x{height}")

# 2. Save to text file
# Format: 
# Line 1: Width Height
# Line 2: pixel pixel pixel ...
with open("img_data.txt", "w") as f:
    f.write(f"{width} {height}\n")
    # Save the array as a long space-separated string
    img.flatten().tofile(f, sep=" ", format="%d")

print("Success! Converted image to 'img_data.txt'")