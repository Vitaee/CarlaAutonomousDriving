import torch
from model import NvidiaModel
from torchvision import transforms
from config import config
import cv2
import numpy as np
# Load the model and set it to eval mode

model_class = NvidiaModel
model = model_class()
model.load_state_dict(torch.load("./save/model.pt", map_location=torch.device(config.device)))
#model.load_state_dict(torch.load("./save_center/model.pt", map_location=torch.device(config.device)))
model.to(config.device)
model.eval()


def crop_down(image):
    """Crop the top portion of the image (remove sky/horizon)"""
    h = image.shape[0]
    w = image.shape[1]
    top = 90
    crop_height = h - top
    return image[top:top + crop_height, :]

def preprocess_image_for_model(image):
    """
    Preprocess image exactly as done in training pipeline.
    Input: RGB image (H, W, C) as numpy array
    Output: Torch tensor ready for model
    """
    # Step 1: Crop the image (remove top portion like sky)
    image_cropped = crop_down(image)
    
    # Step 2: Convert RGB to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
    
    # Step 3: Convert BGR to YUV (exactly like training)
    image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    
    # Step 4: Resize to match training size (width=200, height=66)
    image_resized = cv2.resize(image_yuv, (200, 66))
    
    # Step 5: Convert to torch tensor and normalize exactly like training
    # Transpose from (H, W, C) to (C, H, W)
    image_tensor = np.transpose(image_resized, (2, 0, 1))
    image_tensor = torch.from_numpy(image_tensor).float()
    
    # Step 6: Normalize to [-1.0, 1.0] exactly like training
    image_tensor = (image_tensor / 127.5) - 1.0
    
    # Step 7: Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(config.device)
    
    return image_tensor

def predict_steering_angle(image):
    """
    Predict steering angle from RGB image.
    Input: RGB image (H, W, C) as numpy array
    Output: Steering angle as float
    """
    # Preprocess image exactly like training
    processed_image = preprocess_image_for_model(image)
    
    # Predict
    with torch.no_grad():
        prediction = model(processed_image)
        steering_angle = prediction.item()
    
    return steering_angle