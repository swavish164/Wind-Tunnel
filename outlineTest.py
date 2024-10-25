import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('f1CarSide.jpg')
#image = cv2.resize(image,(1000,500))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
kernel = np.ones((5, 5), np.uint8)  # Define a 5x5 kernel
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_image = np.zeros_like(image)
print(np.shape(filled_image))
cv2.drawContours(filled_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
plt.show()