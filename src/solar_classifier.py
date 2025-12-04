import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np


class SolarSystemClassifier:
    """
    Wrapper around the Solar System ResNet-18 model.
    Example classes: ['earth', 'jupiter', 'mars', 'moon', ...]
    """
    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.class_names = checkpoint["class_names"]

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features,
                                  len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, img_rgb: np.ndarray):
        """
        img_rgb: H x W x 3 uint8 (OpenCV BGR→RGB converted image).

        Returns:
            (label_name: str, confidence: float, probs: np.ndarray)
        """
        x = self.tfms(img_rgb).unsqueeze(0)  # 1 x 3 x 224 x 224

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(probs.argmax())
        return self.class_names[idx], float(probs[idx]), probs