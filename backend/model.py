"""CheXNet model with Grad-CAM heatmap generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import cv2

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


class GradCAM:
    """Grad-CAM implementation for DenseNet121-based CheXNet."""

    def __init__(self, model: CheXNet):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Hook into the last convolutional layer of DenseNet features
        target_layer = self.model.densenet.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for a specific class."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Backpropagate for the target class
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        # Pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to input image size
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam


def generate_heatmap_overlay(original_image: Image.Image, cam: np.ndarray) -> str:
    """Overlay Grad-CAM heatmap on the original image. Returns base64 encoded PNG."""
    # Resize original to match
    img_resized = original_image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    img_array = np.array(img_resized)

    # Create heatmap with colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend: 60% original, 40% heatmap
    overlay = np.uint8(0.6 * img_array + 0.4 * heatmap)

    # Encode to base64
    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_standalone_heatmap(cam: np.ndarray) -> str:
    """Generate standalone heatmap image as base64 PNG."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(heatmap)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def predict_with_heatmap(model: CheXNet, image_bytes: bytes, device: torch.device) -> dict:
    """Run inference and generate Grad-CAM heatmap for top predicted class."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    # Enable gradients for Grad-CAM
    img_tensor.requires_grad_(True)

    # Forward pass for predictions
    with torch.enable_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

    # Get predictions
    results = []
    for i, disease in enumerate(DISEASE_LABELS):
        results.append({
            "disease": disease,
            "probability": round(float(probs[i]), 4),
        })
    results.sort(key=lambda x: x["probability"], reverse=True)

    # Top predicted class index
    top_class_idx = int(np.argmax(probs))
    top_disease = DISEASE_LABELS[top_class_idx]

    # Generate Grad-CAM
    grad_cam = GradCAM(model)
    cam = grad_cam.generate(img_tensor, top_class_idx)

    # Generate heatmap images
    heatmap_overlay = generate_heatmap_overlay(img, cam)
    heatmap_standalone = generate_standalone_heatmap(cam)

    # Original image as base64
    buffer = io.BytesIO()
    img.resize((IMG_SIZE, IMG_SIZE)).save(buffer, format="PNG")
    buffer.seek(0)
    original_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "predictions": results,
        "top_prediction": top_disease,
        "top_probability": round(float(probs[top_class_idx]), 4),
        "heatmap_overlay": heatmap_overlay,
        "heatmap_standalone": heatmap_standalone,
        "original_image": original_b64,
        "heatmap_class": top_disease,
    }
