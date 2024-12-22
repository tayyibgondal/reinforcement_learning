import cv2

def preprocess_frame(frame):
    """Preprocesses a single frame: grayscale, resize, normalize."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # Resize to 84x84
    frame = frame / 255.0  # Normalize pixel values
    return frame
