
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
from torchvision.models import EfficientNet_V2_S_Weights


# -----------------------------
# Configuration
# -----------------------------
CLASS_NAMES = [
    "Pothole Issues",
    "Damaged Road issues",
    "Mixed Issues"
]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "efficientnet_v2_s_best.pth"


# -----------------------------
# Model
# -----------------------------
def build_efficientnet_v2_s(num_classes=3, dropout=0.30):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=False),
        nn.Linear(in_features, num_classes)
    )
    return model


@st.cache_resource
def load_model():
    model = build_efficientnet_v2_s(num_classes=len(CLASS_NAMES))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def preprocess_image(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)


def denormalize_image(tensor):
    image = tensor.detach().cpu().clone().squeeze(0)
    for c in range(3):
        image[c] = image[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(self.forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_heatmap(image_np, heatmap, alpha=0.45):
    heatmap_rgb = np.zeros_like(image_np)
    heatmap_rgb[..., 0] = heatmap
    heatmap_rgb[..., 1] = heatmap * 0.3
    heatmap_rgb[..., 2] = 1 - heatmap
    overlay = (1 - alpha) * image_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay


# -----------------------------
# Prediction
# -----------------------------
def predict(model, image: Image.Image):
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3_df = pd.DataFrame({
        "Class": [CLASS_NAMES[i] for i in top3_idx],
        "Confidence": [float(probs[i]) for i in top3_idx]
    })

    return input_tensor, pred_idx, pred_label, confidence, top3_df


def generate_gradcam(model, input_tensor, pred_idx):
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    heatmap, _ = gradcam.generate(input_tensor, class_idx=pred_idx)
    gradcam.close()

    original_np = denormalize_image(input_tensor)
    overlay_np = overlay_heatmap(original_np, heatmap)
    return original_np, heatmap, overlay_np


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Road Issues Detection", layout="wide")

st.title("Road Issues Detection using Deep Learning")
st.write("Upload a road image to classify it into Pothole Issues, Damaged Road issues, or Mixed Issues.")
st.write("This demo uses the best-performing model: **EfficientNetV2-S**.")

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "webp"]
)

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        input_tensor, pred_idx, pred_label, confidence, top3_df = predict(model, image)
        original_np, heatmap, overlay_np = generate_gradcam(model, input_tensor, pred_idx)

        with col2:
            st.success(f"Prediction: {pred_label}")
            st.info(f"Confidence: {confidence:.4f}")

            st.subheader("Top 3 Predictions")
            st.dataframe(top3_df, use_container_width=True)

        st.subheader("Grad-CAM Explanation")
        g1, g2, g3 = st.columns(3)

        with g1:
            st.image(original_np, caption="Preprocessed Image", use_container_width=True)

        with g2:
            st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)

        with g3:
            st.image(overlay_np, caption="Overlay", use_container_width=True)

else:
    st.caption("Please upload an image to begin.")


