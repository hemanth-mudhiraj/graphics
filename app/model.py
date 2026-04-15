"""Model loading and inference for CheXNet chest X-ray classification."""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

IMG_SIZE = 224

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class CheXNet(nn.Module):
    """DenseNet121-based model for 14-class multi-label chest X-ray classification."""

    def __init__(self, num_classes=14):
        super(CheXNet, self).__init__()
        self.densenet = models.densenet121(weights=None)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.densenet(x)


def load_model(model_path: str, device: torch.device) -> CheXNet:
    """Load the trained CheXNet model from a checkpoint."""
    model = CheXNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def predict(model: CheXNet, image_bytes: bytes, device: torch.device) -> list[dict]:
    """Run inference on an uploaded image. Returns list of {disease, probability}."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = []
    for i, disease in enumerate(DISEASE_LABELS):
        results.append({
            "disease": disease,
            "probability": round(float(probs[i]), 4),
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results
