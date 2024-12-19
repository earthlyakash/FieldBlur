import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Step 1: Upload your image
uploaded = files.upload()
input_file = next(iter(uploaded))

# Load the image
image = cv2.imread(input_file, cv2.IMREAD_COLOR)
height, width = image.shape[:2]

# Step 2: Define a function for custom blur mask
def create_blur_mask(locations, intensities, radius=150):
    """
    Creates a blur mask with customizable blur locations and intensities.
    Args:
    - locations: List of (x, y) tuples specifying blur points.
    - intensities: List of blur intensity values corresponding to each point.
    - radius: Radius around each point for blur influence.
    Returns:
    - A blur mask with smooth transitions.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for (x, y), intensity in zip(locations, intensities):
        # Draw a circle with the given intensity
        cv2.circle(mask, (x, y), radius, intensity, -1, cv2.LINE_AA)
    # Smooth transitions between points
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    return mask

# Step 3: Define locations and intensities
# Example: Specify blur points and their corresponding intensities
locations = [
    (width // 4, height // 4),       # Top-left corner
    (width // 2, height // 2),       # Center
    (3 * width // 4, 3 * height // 4) # Bottom-right corner
]

intensities = [
    0.2,  # Low blur intensity
    0.8,  # High blur intensity
    0.5   # Medium blur intensity
]

# Ensure locations and intensities lists are the same length
assert len(locations) == len(intensities), "Locations and intensities must have the same length!"

# Create the blur mask
blur_mask = create_blur_mask(locations, intensities, radius=200)

# Step 4: Apply per-pixel blur based on the mask
blurred_image = np.zeros_like(image)
for i in range(3):  # Process each channel (R, G, B)
    # Apply a full blur to the channel
    fully_blurred = cv2.GaussianBlur(image[:, :, i], (51, 51), 0)
    # Blend using the mask: mask * blurred + (1 - mask) * original
    blurred_image[:, :, i] = (blur_mask * fully_blurred + (1 - blur_mask) * image[:, :, i]).astype(np.uint8)

# Step 5: Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Blur Mask")
plt.imshow(blur_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Field Blur Result")
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()

# Step 6: Save and download the result
output_file = "field_blur_result.png"
cv2.imwrite(output_file, blurred_image)
files.download(output_file)
