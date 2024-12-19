import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

def field_blur_with_lists(image, locations, blur_values, resize_if_large=True):
    """
    Apply field blur with optimization for large images
    """
    # Check image size and resize if too large
    height, width = image.shape[:2]
    original_size = (width, height)
    
    if resize_if_large and (width > 1000 or height > 1000):
        # Calculate new size while maintaining aspect ratio
        scale = 1000 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        img = Image.fromarray(image)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image = np.array(img)
        
        # Scale locations accordingly
        locations = [(int(x * scale), int(y * scale)) for x, y in locations]
        blur_values = [(int(r * scale), s) for r, s in blur_values]
        
        print(f"Resized image from {original_size} to {new_width}x{new_height} for faster processing")
    
    # Create blur map
    height, width = image.shape[:2]
    blur_map = np.zeros((height, width))
    
    # Process each blur point with progress bar
    print("Calculating blur map...")
    for (x, y), (radius, strength) in tqdm(zip(locations, blur_values)):
        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        influence = np.clip(1 - distances/radius, 0, 1) * strength
        blur_map = np.maximum(blur_map, influence)
    
    # Create output image
    result = np.zeros_like(image, dtype=float)
    max_radius = max(point[0] for point in blur_values)
    
    # Apply blur with progress bar
    print("Applying blur effect...")
    unique_strengths = np.unique(blur_map)
    for strength in tqdm(unique_strengths):
        if strength > 0:
            mask = blur_map == strength
            sigma = strength * max_radius / 3
            if len(image.shape) == 2:
                blurred = gaussian_filter(image, sigma=sigma)
            else:
                blurred = gaussian_filter(image, sigma=(sigma, sigma, 0))
            result[mask] = blurred[mask]
    
    # Copy unblurred regions
    mask = blur_map == 0
    result[mask] = image[mask]
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Resize back to original size if we resized earlier
    if resize_if_large and (original_size[0] > 1000 or original_size[1] > 1000):
        img = Image.fromarray(result)
        img = img.resize(original_size, Image.Resampling.LANCZOS)
        result = np.array(img)
    
    return result

def apply_field_blur(image_path, locations, blur_values, save_path=None):
    """
    Apply field blur with progress tracking and optimization
    """
    print("Loading image...")
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print("Applying field blur...")
    result = field_blur_with_lists(img_array, locations, blur_values)
    
    # Save result first (before displaying)
    if save_path:
        print(f"Saving result to {save_path}...")
        Image.fromarray(result).save(save_path)
        print("Save completed!")
    
    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray' if len(result.shape) == 2 else None)
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.show()
    return result

# Example usage
print("Please upload an image...")
from google.colab import files
uploaded = files.upload()

image_path = next(iter(uploaded))

# Automatically calculate good points based on image size
img = Image.open(image_path)
width, height = img.size

# Create a 3x3 grid of blur points
locations = [
    (width//4, height//4),    # Top left
    (width//2, height//4),    # Top center
    (3*width//4, height//4),  # Top right
    (width//4, height//2),    # Middle left
    (width//2, height//2),    # Center
    (3*width//4, height//2),  # Middle right
    (width//4, 3*height//4),  # Bottom left
    (width//2, 3*height//4),  # Bottom center
    (3*width//4, 3*height//4) # Bottom right
]

# Create corresponding blur values
blur_values = [
    (width//8, 0.7),  # Radius is 1/8 of image width
    (width//8, 0.7),
    (width//8, 0.7),
    (width//8, 0.7),
    (width//8, 0.7),
    (width//8, 0.7),
    (width//8, 0.7),
    (width//8, 0.7),
    (width//8, 0.7)
]

print("Processing image...")
result = apply_field_blur(image_path, locations, blur_values, 'blurred_output.jpg')

print("Downloading result...")
files.download('blurred_output.jpg')
