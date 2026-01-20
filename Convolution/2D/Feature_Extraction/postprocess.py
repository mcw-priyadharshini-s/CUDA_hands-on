import cv2
import numpy as np
import sys

# 1. Read the text file
try:
    with open("out_data.txt", "r") as f:
        # Read Header
        header = f.readline().split()
        width = int(header[0])
        height = int(header[1])
        
        # Read Data
        data_str = f.read().split()
        
        # Convert strings to numpy array of bytes (uint8)
        data = np.array(data_str, dtype=np.uint8)
except FileNotFoundError:
    print("Error: Run the CUDA program first!")
    sys.exit()

# 2. Reshape into an Image
output_img = data.reshape((height, width))

# 3. Save
cv2.imwrite("final_output.jpg", output_img)
print(f"Success! Image saved as 'final_output.jpg' ({width}x{height})")