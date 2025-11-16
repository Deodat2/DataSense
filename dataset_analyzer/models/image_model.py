# dataset_analyzer/image_model.py

# Import necessary libraries
from PIL import Image  # Python Imaging Library, used to open and manipulate images
import torch  # PyTorch library for deep learning
from torchvision import models, transforms  # Pre-trained models and image transformations

# -----------------------------------------------------------
# 1. Load a pre-trained ResNet model
# -----------------------------------------------------------

# ResNet18 is a convolutional neural network pre-trained on ImageNet (a large image dataset)
# This model can classify images into 1000 object categories (like dog, cat, car, etc.)
resnet = models.resnet18(pretrained=True)

# Set the model to evaluation mode
# This disables layers like dropout and ensures the model behaves correctly for inference
resnet.eval()

# -----------------------------------------------------------
# 2. Define preprocessing steps
# -----------------------------------------------------------

# Preprocessing is necessary to convert raw images into the format expected by the model
# Typical steps include resizing, cropping, normalization, and converting to a tensor
preprocess = transforms.Compose([
    transforms.Resize(256),         # Resize the shorter side of the image to 256 pixels
    transforms.CenterCrop(224),     # Crop the center 224x224 region (ResNet expects 224x224 input)
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor (shape: CxHxW)
    transforms.Normalize(           # Normalize using ImageNet mean and std values
        mean=[0.485, 0.456, 0.406], # Mean for each color channel (R,G,B)
        std=[0.229, 0.224, 0.225]   # Standard deviation for each channel
    ),
])


# -----------------------------------------------------------
# 3. Load ImageNet class labels
# -----------------------------------------------------------

# These labels correspond to the 1000 classes the ResNet18 model predicts
imagenet_labels = []
with open("dataset_analyzer/models/imagenet_classes.txt") as f:
    imagenet_labels = [line.strip() for line in f.readlines()]


# -----------------------------------------------------------
# 4. Define the image classification function
# -----------------------------------------------------------

def detect_image_class(image_path):
    """
        Predicts the class of an image using the pre-trained ResNet18 model.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Predicted class (e.g., 'Classe 207') or 'Indéterminée' in case of an error
    """
    try:
        # Step 1: Open the image and convert to RGB (ensure 3 color channels)
        img = Image.open(image_path).convert('RGB')

        # Step 2: Apply preprocessing transforms
        # This prepares the image tensor in the shape (1, 3, 224, 224)
        input_tensor = preprocess(img).unsqueeze(0)

        # Step 3: Forward pass through the model (inference)
        with torch.no_grad():  # Disable gradient calculation (we don't need it for inference)
            outputs = resnet(input_tensor)  # The output is a tensor of size [1, 1000]

        # Step 4: Get the class with the highest predicted score
        _, predicted_idx = outputs.max(1)  # 'predicted' is a tensor containing the class index

        # Step 5: Return the human-readable ImageNet label
        return imagenet_labels[predicted_idx.item()]

    except Exception as e:
        # If anything goes wrong (file not found, wrong format, etc.), return "Indéterminée"
        return "Indéterminée"
