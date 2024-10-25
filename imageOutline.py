import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from PIL import Image

# Load the original image and create shifted versions
image = Image.open('f1CarSide.jpg')
image = image.convert('RGB')

imageShiftLeft = Image.new('RGB', image.size)
imageShiftRight = Image.new('RGB', image.size)
imageShiftLeft.paste(image, (10, 0))  # Shift left
imageShiftRight.paste(image, (15, 0))  # Shift right

# Convert images to numpy arrays
imgLeft = np.array(imageShiftLeft).astype(np.uint8)
imgRight = np.array(imageShiftRight).astype(np.uint8)

# Convert to grayscale
imgLeftGray = cv.cvtColor(imgLeft, cv.COLOR_RGB2GRAY)
imgRightGray = cv.cvtColor(imgRight, cv.COLOR_RGB2GRAY)

# Apply Gaussian blur to smooth the images
imgLeftGray = cv.GaussianBlur(imgLeftGray, (5, 5), 0)
imgRightGray = cv.GaussianBlur(imgRightGray, (5, 5), 0)

# Compute disparity map using StereoSGBM
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=260,  # Adjusted based on the image
    blockSize=5,  # Block size for matching
    P1=8 * 3 * 5 ** 2,  # Parameter for disparity smoothness
    P2=32 * 3 * 5 ** 2,  # Larger value for larger smoothness
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)

# Compute disparity map
disparity = stereo.compute(imgLeftGray, imgRightGray).astype(np.float32) / 16.0  # Scale down

# Normalize the disparity map for better visualization
disparity_normalized = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Optional: Apply bilateral filter to smooth the disparity map
disparity_smoothed = cv.bilateralFilter(disparity_normalized, d=5, sigmaColor=75, sigmaSpace=75)

# Create a heatmap of the object (based on distance)
heatmap = cv.applyColorMap(disparity_smoothed, cv.COLORMAP_JET)

# Thresholding to separate object from background based on disparity
# Using a fixed threshold here, but can also be adjusted
_, object_mask = cv.threshold(disparity_smoothed, 40, 255, cv.THRESH_BINARY)

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)  # Define a kernel for morphological operations
morph_mask = cv.morphologyEx(object_mask, cv.MORPH_CLOSE, kernel)  # Close small holes

# Use the mask to extract only the object from the original image
segmented_object = cv.bitwise_and(imgLeft, imgLeft, mask=morph_mask)

# Display Results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(cv.cvtColor(imgLeft, cv.COLOR_BGR2RGB))

# Heatmap of the disparity (object depth)
plt.subplot(1, 4, 2)
plt.title('Disparity Heatmap')
plt.imshow(cv.cvtColor(heatmap, cv.COLOR_BGR2RGB))

# Smoothed Disparity
plt.subplot(1, 4, 3)
plt.title('Smoothed Disparity Map')
plt.imshow(disparity_smoothed, cmap='gray')

# Segmented Object
plt.subplot(1, 4, 4)
plt.title('Segmented Object')
plt.imshow(cv.cvtColor(segmented_object, cv.COLOR_BGR2RGB))

plt.show()
