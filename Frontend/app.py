
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Road Issues Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "efficientnet_v2_s_best.pth"   # change this only if needed

CLASS_NAMES = [
    "Pothole Issues",
    "Damaged Road issues",
    "Mixed Issues",
]

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 48%, #0ea5e9 100%);
        padding: 30px;
        border-radius: 24px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.96;
        line-height: 1.7;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        text-align: center;
    }

    .metric-title {
        font-size: 0.92rem;
        color: #64748b;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 12px;
        color: #0f172a;
    }

    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .prediction-box {
        background: #ecfeff;
        border: 1px solid #bae6fd;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .soft-note {
        color: #475569;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .footer-box {
        margin-top: 28px;
        padding: 18px;
        background: #f8fafc;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("⚙️ App Information")
    st.markdown("**Model:** EfficientNetV2-S")
    st.markdown("**Task:** 3-Class Road Issue Classification")
    st.markdown("**Input Size:** 224 × 224")
    st.markdown("**Framework:** PyTorch + Streamlit")
    st.divider()

    st.markdown("### Supported Classes")
    st.markdown("- Pothole Issues")
    st.markdown("- Damaged Road issues")
    st.markdown("- Mixed Issues")
    st.divider()

    st.markdown("### Tips")
    st.markdown(
        """
        - Upload a clear road image  
        - JPG / PNG / WEBP supported  
        - Best results come from visible road-surface defects  
        - Large, blurry, or dark images may reduce confidence
        """
    )

# =========================================================
# HERO SECTION
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🛣️ Road Issues Detection using Deep Learning</div>
        <div class="hero-subtitle">
            Upload a road image to classify it into <b>Pothole Issues</b>,
            <b>Damaged Road issues</b>, or <b>Mixed Issues</b>.<br>
            This demo uses the best-performing model: <b>EfficientNetV2-S</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL
# =========================================================
def build_model() -> nn.Module:
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


@st.cache_resource
def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================================================
# PREDICTION
# =========================================================
def predict_image(model: nn.Module, image: Image.Image) -> Tuple[str, float, np.ndarray]:
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs


# =========================================================
# PLACEHOLDER EXPLAINABILITY
# Replace with your real Grad-CAM later if needed
# =========================================================
def generate_placeholder_overlay(image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0

    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    center_y, center_x = h * 0.62, w * 0.55
    dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    dist = dist / (dist.max() + 1e-8)

    heat = 1.0 - dist
    heat = np.clip(heat, 0, 1)

    # build colored heatmap
    heat_rgb = np.zeros((h, w, 3), dtype=np.float32)
    heat_rgb[..., 0] = heat
    heat_rgb[..., 1] = heat * 0.45
    heatmap_img = (heat_rgb * 255).astype(np.uint8)

    overlay = 0.62 * arr + 0.38 * heat_rgb
    overlay = np.clip(overlay, 0, 1)
    overlay_img = (overlay * 255).astype(np.uint8)

    return Image.fromarray(heatmap_img), Image.fromarray(overlay_img)


# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    model = load_model()
    pred_class, confidence, probs = predict_image(model, image)

    top_indices = np.argsort(probs)[::-1]
    top_df = pd.DataFrame({
        "Class": [CLASS_NAMES[i] for i in top_indices],
        "Confidence": [float(probs[i]) for i in top_indices],
    })

    # =====================================================
    # TOP METRICS
    # =====================================================
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Class</div>
                <div class="metric-value" style="font-size:1.18rem;">{pred_class}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Confidence</div>
                <div class="metric-value">{confidence:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Best Model</div>
                <div class="metric-value" style="font-size:1.18rem;">EfficientNetV2-S</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Prediction Overview</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 0.95])

    with left_col:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with right_col:
        st.markdown(
            f"""
            <div class="prediction-box">
                <b>Prediction:</b> {pred_class}<br><br>
                <b>Confidence Score:</b> {confidence:.4f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='soft-note'>Top class probabilities</div>", unsafe_allow_html=True)
        st.dataframe(
            top_df.style.format({"Confidence": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(top_df["Class"], top_df["Confidence"])
        ax.set_ylabel("Confidence")
        ax.set_title("Class Probability Comparison")
        plt.xticks(rotation=12)
        st.pyplot(fig)
        plt.close(fig)

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Explainability", "Model Details", "How to Read Results"])

    with tab1:
        st.markdown("<div class='section-title'>Grad-CAM Explanation</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='soft-note'>This placeholder heatmap keeps the interface polished. You can replace it later with your exact Grad-CAM output.</div>",
            unsafe_allow_html=True,
        )

        heatmap_img, overlay_img = generate_placeholder_overlay(image)

        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(image.resize((IMAGE_SIZE, IMAGE_SIZE)), caption="Preprocessed Image", use_container_width=True)
        with g2:
            st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
        with g3:
            st.image(overlay_img, caption="Overlay", use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Model Details</div>", unsafe_allow_html=True)

        info1, info2 = st.columns(2)
        with info1:
            st.markdown(
                """
                <div class="info-box">
                    <b>Architecture:</b> EfficientNetV2-S<br>
                    <b>Task:</b> 3-class road issue classification<br>
                    <b>Input Size:</b> 224 × 224<br>
                    <b>Deployment:</b> Streamlit
                </div>
                """,
                unsafe_allow_html=True,
            )

        with info2:
            st.markdown(
                """
                <div class="info-box">
                    <b>Output Classes:</b><br>
                    1. Pothole Issues<br>
                    2. Damaged Road issues<br>
                    3. Mixed Issues
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown("<div class='section-title'>How to Interpret the Output</div>", unsafe_allow_html=True)
        st.markdown(
            """
            - **Predicted Class** is the model's most likely class.  
            - **Confidence Score** is the probability assigned to that class.  
            - **Top Predictions** show how strongly the model considered the other classes.  
            - **Grad-CAM** highlights which road region most influenced the prediction.
            """
        )

else:
    st.markdown(
        """
        <div class="info-box">
            Upload a road image to begin classification.  
            The app will display the predicted class, confidence score, class probabilities,
            and an explanation view.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer-box">
        <b>Project:</b> Explainable Transfer Learning for Road Damage and Surface Defect Classification in Smart City Infrastructure Monitoring<br>
        Built with Streamlit, PyTorch, EfficientNetV2-S, and explainability support.
    </div>
    """,
    unsafe_allow_html=True,
)
