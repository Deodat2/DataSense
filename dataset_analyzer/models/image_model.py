# dataset_analyzer/image_model.py

# Import necessary libraries
import logging
import warnings
from typing import Optional, List, Any

try:
    from PIL import Image  # Python Imaging Library, used to open and manipulate images
    import torch  # PyTorch library for deep learning
    from torchvision import models, transforms  # Pre-trained models and image transformations
    import importlib.resources as pkg_resources
    _IMAGE_DEPS_AVAILABLE = True
except ImportError:
    _IMAGE_DEPS_AVAILABLE = False


logger = logging.getLogger("DataSense.ImageModel")

warnings.filterwarnings("ignore", category=UserWarning)
_RESOURCE_PACKAGE = 'dataset_analyzer.models'


class ImageClassifierService:
    """
    Manages the lazy loading and inference of the ImageNet classifier (ResNet18).
    """

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._labels: Optional[List[str]] = None

    def _load_resources(self):
        """Loads the model, transforms, and labels only once."""
        if not _IMAGE_DEPS_AVAILABLE:
            logger.warning(
                "Skipping Image Analysis: PyTorch, torchvision, or Pillow not found. "
                "Install them to enable image analysis."
            )
            return False

        if self._model is not None:
            return True  # Already loaded

        logger.info("Initializing ResNet18 classifier (lazy load)...")
        try:
            # 1. Load Labels from package resources
            # Utilisez le chemin du package au lieu d'un chemin de fichier relatif
            labels_content = pkg_resources.read_text(_RESOURCE_PACKAGE, 'imagenet_classes.txt')
            self._labels = [line.strip() for line in labels_content.splitlines()]

            # 2. Load Model
            self._model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self._model.eval()

            # 3. Define Preprocessing
            # Preprocessing is necessary to convert raw images into the format expected by the model
            # Typical steps include resizing, cropping, normalization, and converting to a tensor
            self._preprocess = transforms.Compose([
                transforms.Resize(256),             # Resize the shorter side of the image to 256 pixels
                transforms.CenterCrop(224),         # Crop the center 224x224 region (ResNet expects 224x224 input)
                transforms.ToTensor(),              # Convert the image to a PyTorch tensor (shape: CxHxW)
                transforms.Normalize(               # Normalize using ImageNet mean and std values
                    mean=[0.485, 0.456, 0.406],     # Mean for each color channel (R,G,B)
                    std=[0.229, 0.224, 0.225]       # Standard deviation for each channel
                ),
            ])
            logger.info("Image Classifier ready.")
            return True

        except Exception as e:
            # Si le téléchargement ou le chargement échoue (ex: mauvaise connexion, permission)
            logger.error("Failed to load image classification resources: %s", e)
            self._model = None
            return False

    def detect_image_class(self, image_path: str) -> str:
        """Predicts the class of an image."""
        if not self._load_resources():
            return "Dependencies missing"

        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = self._preprocess(img).unsqueeze(0)

            with torch.no_grad():
                outputs = self._model(input_tensor)

            _, predicted_idx = outputs.max(1)
            return self._labels[predicted_idx.item()]

        except Exception as e:
            logger.debug("Image analysis error for %s: %s", image_path, e)
            return "Indéterminée"

# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------

# Créez une instance du service une seule fois
_IMAGE_CLASSIFIER_SERVICE = ImageClassifierService()

def detect_image_class(image_path: str) -> str:
    """
    Public entry point for image classification, using lazy loading service.
    """
    return _IMAGE_CLASSIFIER_SERVICE.detect_image_class(image_path)

