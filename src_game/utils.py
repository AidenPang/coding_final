import cv2
import numpy as np

def preprocess_canvas_image(canvas_img):
    """
    Processes a high-resolution drawing canvas image into a 28x28 normalized 
    image suitable for MNIST digit classification, centering the digit.
    
    Parameters:
        canvas_img (np.ndarray): Grayscale or binary image of the canvas, 
                                 where drawing is bright (255) and background is dark (0).
                                 Shape can be (H, W).
                                 
    Returns:
        np.ndarray: Preprocessed 28x28 image (values in range [0.0, 1.0]).
    """
    # 1. Convert to grayscale if it has color channels
    if len(canvas_img.shape) == 3:
        if canvas_img.shape[2] == 4: # RGBA
            # Extract alpha channel or convert to grayscale
            canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGBA2GRAY)
        elif canvas_img.shape[2] == 3: # RGB/BGR
            canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            
    # 2. Threshold image to make it binary (ensure black background, white drawing)
    _, thresh = cv2.threshold(canvas_img, 20, 255, cv2.THRESH_BINARY)
    
    # 3. Find bounding box of the drawn digit
    # Find all non-zero coordinate points
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        # Canvas is completely empty, return a blank 28x28 image
        return np.zeros((28, 28), dtype=np.float32)
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop the digit from the thresh image
    digit_crop = thresh[y:y+h, x:x+w]
    
    # 4. Resize cropped digit to fit within a 20x20 box (maintaining aspect ratio)
    # MNIST requires the digit to be placed inside a 20x20 bounding box 
    # to maintain standard padding.
    max_dim = max(w, h)
    
    # Scale factor to make max dimension 20 pixels
    scale = 20.0 / max_dim
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    # Ensure dimensions are at least 1 pixel
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    # Resize using area interpolation (good for shrinking)
    digit_resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 5. Place the 20x20 (or smaller) digit into the center of a black 28x28 canvas
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate offset to center the resized digit
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    
    mnist_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = digit_resized
    
    # 6. Dilate slightly to ensure stroke thickness matches MNIST style (improves recognition of thin lines & '7')
    kernel = np.ones((2, 2), np.uint8)
    mnist_canvas = cv2.dilate(mnist_canvas, kernel, iterations=1)
    
    # 7. Apply slight Gaussian Blur to mimic MNIST dataset's soft stroke edges
    # This greatly increases model recognition accuracy for hand-drawn inputs
    mnist_canvas = cv2.GaussianBlur(mnist_canvas, (3, 3), 0)
    
    # 7. Normalize to range [0.0, 1.0]
    normalized_img = mnist_canvas.astype(np.float32) / 255.0
    
    return normalized_img
