import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import time
import io

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Damage Classifier",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "efficientnet_v2_s_best.pth"

CLASS_NAMES = ["Damaged Road issues", "Mixed Issues", "Pothole Issues"]
CLASS_ICONS = {"Damaged Road issues": "🔧", "Mixed Issues": "⚠️", "Pothole Issues": "🕳️"}
CLASS_COLORS = {"Damaged Road issues": "#e67e22", "Mixed Issues": "#9b59b6", "Pothole Issues": "#e74c3c"}
CLASS_DESCRIPTIONS = {
    "Pothole Issues": "Localised depressions or cavities in the road surface, typically caused by water infiltration and traffic wear.",
    "Damaged Road issues": "Broader structural deterioration such as cracking, broken asphalt, fragmented pavement, or worn surface regions.",
    "Mixed Issues": "Multiple co-occurring defect types present in the same road section, making the condition more complex.",
}

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: linear-gradient(180deg, #0f1724 0%, #1a2332 100%);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e2e 0%, #0d1520 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: #c8d6e5 !important;
}

/* Hero banner */
.hero-container {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d1b4e 50%, #1a3347 100%);
    border-radius: 16px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 32px;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(52,152,219,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #5dade2;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 32px;
    font-weight: 800;
    color: #ecf0f1;
    line-height: 1.2;
    margin-bottom: 12px;
}
.hero-subtitle {
    font-size: 15px;
    font-weight: 400;
    color: #8e9eb3;
    line-height: 1.6;
    max-width: 700px;
}

/* Stat pills in hero */
.hero-stats {
    display: flex;
    gap: 12px;
    margin-top: 24px;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 500;
    color: #aab7c9;
}
.hero-pill strong {
    color: #5dade2;
    font-weight: 700;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}
.glass-card-title {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #5dade2;
    margin-bottom: 16px;
}

/* Prediction result card */
.prediction-card {
    background: linear-gradient(135deg, rgba(46,204,113,0.08) 0%, rgba(39,174,96,0.04) 100%);
    border: 1px solid rgba(46,204,113,0.2);
    border-radius: 14px;
    padding: 32px;
    text-align: center;
    margin-bottom: 20px;
}
.prediction-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2ecc71;
    margin-bottom: 8px;
}
.prediction-class {
    font-size: 28px;
    font-weight: 800;
    color: #ecf0f1;
    margin-bottom: 4px;
}
.prediction-confidence {
    font-size: 42px;
    font-weight: 800;
    color: #2ecc71;
    margin-bottom: 4px;
}
.prediction-subtext {
    font-size: 13px;
    color: #8e9eb3;
}

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 12px;
}
.prob-label {
    font-size: 13px;
    font-weight: 500;
    color: #c8d6e5;
    min-width: 150px;
    text-align: right;
}
.prob-bar-bg {
    flex: 1;
    height: 28px;
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    display: flex;
    align-items: center;
    padding-left: 10px;
    font-size: 12px;
    font-weight: 700;
    color: white;
    transition: width 0.6s ease;
}
.prob-bar-fill.pothole { background: linear-gradient(90deg, #e74c3c, #c0392b); }
.prob-bar-fill.damaged { background: linear-gradient(90deg, #e67e22, #d35400); }
.prob-bar-fill.mixed   { background: linear-gradient(90deg, #9b59b6, #8e44ad); }

/* Model comparison table */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.comparison-table thead th {
    background: rgba(93,173,226,0.12);
    color: #5dade2;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 14px 16px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.comparison-table tbody td {
    padding: 14px 16px;
    font-size: 14px;
    color: #c8d6e5;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    background: rgba(255,255,255,0.02);
}
.comparison-table tbody tr:hover td {
    background: rgba(93,173,226,0.06);
}
.comparison-table .best-val {
    color: #2ecc71;
    font-weight: 700;
}
.comparison-table .model-name {
    font-weight: 600;
    color: #ecf0f1;
}
.comparison-table .winner-row td {
    background: rgba(46,204,113,0.06);
    border-left: 3px solid #2ecc71;
}

/* Class info cards */
.class-info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px;
    height: 100%;
}
.class-info-icon {
    font-size: 32px;
    margin-bottom: 8px;
}
.class-info-name {
    font-size: 16px;
    font-weight: 700;
    color: #ecf0f1;
    margin-bottom: 6px;
}
.class-info-desc {
    font-size: 13px;
    color: #8e9eb3;
    line-height: 1.5;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8e9eb3;
    font-weight: 500;
    font-size: 14px;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: rgba(93,173,226,0.15) !important;
    color: #5dade2 !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 24px;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(93,173,226,0.3) !important;
    border-radius: 12px !important;
    background: rgba(93,173,226,0.04) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(93,173,226,0.5) !important;
    background: rgba(93,173,226,0.08) !important;
}

/* Metric section */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 22px;
    font-weight: 800;
    color: #5dade2;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6b7d93;
    margin-top: 4px;
}

/* Footer */
.app-footer {
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 24px 32px;
    margin-top: 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.footer-left {
    font-size: 13px;
    color: #6b7d93;
}
.footer-left strong {
    color: #aab7c9;
}
.footer-right {
    display: flex;
    gap: 20px;
}
.footer-link {
    font-size: 12px;
    color: #5dade2;
    text-decoration: none;
    font-weight: 500;
}
.footer-link:hover {
    color: #85c1e9;
}

/* Sidebar card */
.sidebar-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5dade2;
    margin-bottom: 10px;
}

/* Image container */
.uploaded-image-container {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 16px;
}

/* Spinner override */
.stSpinner > div {
    border-top-color: #5dade2 !important;
}

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(93,173,226,0.2), transparent);
    margin: 32px 0;
}

/* Explainability placeholder */
.xai-card {
    background: linear-gradient(135deg, rgba(155,89,182,0.08) 0%, rgba(52,152,219,0.08) 100%);
    border: 1px solid rgba(155,89,182,0.15);
    border-radius: 14px;
    padding: 32px;
    text-align: center;
}
.xai-icon { font-size: 48px; margin-bottom: 12px; }
.xai-title { font-size: 18px; font-weight: 700; color: #ecf0f1; margin-bottom: 8px; }
.xai-text { font-size: 14px; color: #8e9eb3; line-height: 1.6; max-width: 500px; margin: 0 auto; }

/* Methodology steps */
.step-item {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    align-items: flex-start;
}
.step-number {
    background: rgba(93,173,226,0.15);
    border: 1px solid rgba(93,173,226,0.3);
    color: #5dade2;
    font-weight: 800;
    font-size: 14px;
    min-width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.step-content {
    flex: 1;
}
.step-title {
    font-size: 15px;
    font-weight: 600;
    color: #ecf0f1;
    margin-bottom: 2px;
}
.step-desc {
    font-size: 13px;
    color: #8e9eb3;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    if MODEL_PATH.exists():
        state = torch.load(str(MODEL_PATH), map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(state)
    else:
        st.error("Model file not found at: " + str(MODEL_PATH))
        return None
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict(model, image):
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)
    start = time.time()
    with torch.no_grad():
        outputs = model(tensor)
    elapsed = (time.time() - start) * 1000
    probs = torch.softmax(outputs, dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return CLASS_NAMES[idx.item()], conf.item(), {CLASS_NAMES[i]: probs[i].item() for i in range(3)}, elapsed


def generate_gradcam(model, image):
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True

    features = []
    grads = []

    def forward_hook(module, inp, out):
        features.append(out)

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        return None

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    output = model(tensor)
    pred_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_idx].backward()

    fh.remove()
    bh.remove()

    if not features or not grads:
        return None

    feat = features[0].detach()
    grad = grads[0].detach()
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)) / 255.0

    img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    heatmap = np.zeros_like(img_array)
    heatmap[:, :, 0] = cam_resized
    heatmap[:, :, 1] = cam_resized * 0.4
    overlay = np.clip(img_array * 0.55 + heatmap * 0.6, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 16px 0;">
        <div style="font-size: 40px; margin-bottom: 4px;">🛣️</div>
        <div style="font-size: 18px; font-weight: 800; color: #ecf0f1;">Road Damage</div>
        <div style="font-size: 14px; font-weight: 500; color: #5dade2;">Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        w, h = image.size
        size_kb = uploaded_file.size / 1024
        st.markdown(f"""
        <div class="sidebar-card">
            <div style="font-size:12px; color:#6b7d93; font-weight:600; margin-bottom:6px;">IMAGE DETAILS</div>
            <div style="font-size:13px; color:#c8d6e5;">{w} x {h} px &nbsp;·&nbsp; {size_kb:.0f} KB</div>
            <div style="font-size:13px; color:#c8d6e5;">{uploaded_file.name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div style="font-size:12px; color:#6b7d93; font-weight:600; margin-bottom:8px;">ACTIVE MODEL</div>
        <div style="font-size:15px; font-weight:700; color:#ecf0f1;">EfficientNetV2-S</div>
        <div style="font-size:12px; color:#8e9eb3; margin-top:4px;">
            98.89% accuracy &nbsp;·&nbsp; 77.85 MB
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div style="font-size:12px; color:#6b7d93; font-weight:600; margin-bottom:8px;">DATASET</div>
        <div style="font-size:13px; color:#c8d6e5;">Road Issues Detection</div>
        <div style="font-size:12px; color:#8e9eb3; margin-top:2px;">
            4,216 images &nbsp;·&nbsp; 3 classes
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-eyebrow">Smart City Infrastructure Monitoring</div>
    <div class="hero-title">Road Damage &amp; Surface Defect Classifier</div>
    <div class="hero-subtitle">
        An explainable deep learning system that classifies road-surface images
        and highlights the regions driving each prediction, built for practical
        maintenance triage.
    </div>
    <div class="hero-stats">
        <div class="hero-pill"><strong>98.89%</strong>&nbsp; Test Accuracy</div>
        <div class="hero-pill"><strong>0.969</strong>&nbsp; Macro F1</div>
        <div class="hero-pill"><strong>3</strong>&nbsp; Classes</div>
        <div class="hero-pill"><strong>4,216</strong>&nbsp; Images</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main tabs ────────────────────────────────────────────────────
tab_predict, tab_explain, tab_compare, tab_about = st.tabs([
    "🔍  Prediction",
    "🧠  Explainability",
    "📊  Model Comparison",
    "📖  About",
])


# ── TAB 1: Prediction ───────────────────────────────────────────
with tab_predict:
    if uploaded_file:
        model = load_model()
        if model is not None:
            with st.spinner("Classifying..."):
                pred_class, confidence, probs, inf_time = predict(model, image)

            col_result, col_probs = st.columns([1, 1], gap="large")

            with col_result:
                icon = CLASS_ICONS.get(pred_class, "")
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted Class</div>
                    <div class="prediction-class">{icon} {pred_class}</div>
                    <div class="prediction-confidence">{confidence*100:.1f}%</div>
                    <div class="prediction-subtext">confidence &nbsp;·&nbsp; {inf_time:.1f} ms inference</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-title">What This Means</div>
                    <p style="font-size:14px; color:#c8d6e5; line-height:1.6;">
                        {CLASS_DESCRIPTIONS.get(pred_class, "")}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_probs:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="glass-card-title">Class Probabilities</div>', unsafe_allow_html=True)

                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for cls_name, prob in sorted_probs:
                    bar_class = "pothole" if "Pothole" in cls_name else ("damaged" if "Damaged" in cls_name else "mixed")
                    pct = prob * 100
                    width = max(pct, 2)
                    icon_c = CLASS_ICONS.get(cls_name, "")
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-label">{icon_c} {cls_name}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill {bar_class}" style="width:{width}%;">
                                {pct:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card" style="margin-top:16px;">
                    <div class="glass-card-title">Inference Details</div>
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-value">{inf_time:.1f}</div>
                            <div class="metric-label">ms latency</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">224</div>
                            <div class="metric-label">input size</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">20.2M</div>
                            <div class="metric-label">parameters</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:60px 32px;">
            <div style="font-size:56px; margin-bottom:16px;">📷</div>
            <div style="font-size:20px; font-weight:700; color:#ecf0f1; margin-bottom:8px;">
                Upload a Road Image
            </div>
            <div style="font-size:14px; color:#8e9eb3; max-width:400px; margin:0 auto; line-height:1.6;">
                Use the sidebar to upload a JPG or PNG image of a road surface.
                The model will classify the damage type and show confidence scores.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card-title" style="margin-top:32px;">Recognised Damage Classes</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (cls_name, desc) in enumerate(CLASS_DESCRIPTIONS.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="class-info-card">
                    <div class="class-info-icon">{CLASS_ICONS[cls_name]}</div>
                    <div class="class-info-name">{cls_name}</div>
                    <div class="class-info-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ── TAB 2: Explainability ───────────────────────────────────────
with tab_explain:
    if uploaded_file:
        model = load_model()
        if model is not None:
            st.markdown("""
            <div class="glass-card">
                <div class="glass-card-title">Grad-CAM Explanation</div>
                <p style="font-size:14px; color:#8e9eb3; line-height:1.6; margin-bottom:16px;">
                    Grad-CAM highlights which image regions most influenced the prediction.
                    Bright red areas had the strongest effect on the model's decision.
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_orig, col_cam = st.columns(2, gap="large")
            with col_orig:
                st.markdown('<div style="font-size:12px; font-weight:600; color:#6b7d93; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">Original Image</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with col_cam:
                st.markdown('<div style="font-size:12px; font-weight:600; color:#6b7d93; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">Grad-CAM Overlay</div>', unsafe_allow_html=True)
                with st.spinner("Generating Grad-CAM..."):
                    gradcam_img = generate_gradcam(model, image)
                if gradcam_img is not None:
                    st.image(gradcam_img, use_container_width=True)
                else:
                    st.info("Could not generate Grad-CAM for this image.")

            st.markdown("""
            <div class="glass-card" style="margin-top:20px;">
                <div class="glass-card-title">How to Read This</div>
                <p style="font-size:14px; color:#c8d6e5; line-height:1.7;">
                    <strong style="color:#e74c3c;">Bright red regions</strong> indicate areas where the model
                    concentrated its attention most strongly when making the prediction.
                    For a trustworthy prediction, these regions should overlap with visible
                    road defects such as potholes, cracks, or surface damage rather than
                    irrelevant background elements like shadows or lane markings.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="xai-card">
            <div class="xai-icon">🧠</div>
            <div class="xai-title">Explainability Visualisations</div>
            <div class="xai-text">
                Upload a road image to see Grad-CAM attention maps that reveal
                which regions of the image the model focuses on when making its prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">Methods Used in This Study</div>
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-content">
                <div class="step-title">Grad-CAM</div>
                <div class="step-desc">
                    Generates class-specific activation maps by combining gradients with
                    feature maps from the last convolutional layer. Highlights where the
                    model looked when predicting each class.
                </div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-content">
                <div class="step-title">SHAP</div>
                <div class="step-desc">
                    Estimates each image region's contribution using cooperative game theory.
                    Shows both positive evidence (supporting the prediction) and negative
                    evidence (arguing against it). Computed offline due to computational cost.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 3: Model Comparison ─────────────────────────────────────
with tab_compare:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">Three-Model Comparison</div>
        <p style="font-size:14px; color:#8e9eb3; line-height:1.6; margin-bottom:20px;">
            All three models were trained on the same 4,216-image dataset with identical
            preprocessing, stratified splitting, and weighted sampling. Results are from
            the held-out 633-image test set.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="comparison-table">
    <thead>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Macro F1</th>
            <th>Weighted F1</th>
            <th>Top-2 Acc</th>
            <th>Inference</th>
            <th>Size</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="model-name">Custom CNN</td>
            <td>49.45%</td>
            <td>0.305</td>
            <td>0.557</td>
            <td>81.83%</td>
            <td>0.69 ms</td>
            <td>0.43 MB</td>
        </tr>
        <tr class="winner-row">
            <td class="model-name">EfficientNetV2-S &nbsp;🏆</td>
            <td class="best-val">98.89%</td>
            <td class="best-val">0.969</td>
            <td class="best-val">0.989</td>
            <td class="best-val">100%</td>
            <td>2.94 ms</td>
            <td>77.85 MB</td>
        </tr>
        <tr>
            <td class="model-name">ConvNeXt-Tiny</td>
            <td>98.26%</td>
            <td>0.947</td>
            <td>0.983</td>
            <td>100%</td>
            <td>3.96 ms</td>
            <td>106.20 MB</td>
        </tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    col_k1, col_k2, col_k3 = st.columns(3)

    with col_k1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:14px; font-weight:600; color:#e74c3c; margin-bottom:6px;">Custom CNN</div>
            <div style="font-size:36px; font-weight:800; color:#e74c3c;">49.5%</div>
            <div style="font-size:12px; color:#8e9eb3;">accuracy · 110K params</div>
            <div style="font-size:12px; color:#6b7d93; margin-top:8px;">
                Lightweight baseline trained from scratch.
                Too weak for practical road inspection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        st.markdown("""
        <div class="glass-card" style="text-align:center; border-color: rgba(46,204,113,0.3);">
            <div style="font-size:14px; font-weight:600; color:#2ecc71; margin-bottom:6px;">EfficientNetV2-S 🏆</div>
            <div style="font-size:36px; font-weight:800; color:#2ecc71;">98.9%</div>
            <div style="font-size:12px; color:#8e9eb3;">accuracy · 20.2M params</div>
            <div style="font-size:12px; color:#6b7d93; margin-top:8px;">
                Best accuracy-efficiency trade-off.
                Selected for deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_k3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:14px; font-weight:600; color:#e67e22; margin-bottom:6px;">ConvNeXt-Tiny</div>
            <div style="font-size:36px; font-weight:800; color:#e67e22;">98.3%</div>
            <div style="font-size:12px; color:#8e9eb3;">accuracy · 27.8M params</div>
            <div style="font-size:12px; color:#6b7d93; margin-top:8px;">
                Strong but larger and slower.
                Too large for Streamlit Cloud.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">Confusion Matrix — EfficientNetV2-S</div>
        <p style="font-size:14px; color:#8e9eb3; margin-bottom:16px;">
            Only 7 out of 633 test images were misclassified. All errors occurred between
            visually similar classes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="comparison-table">
    <thead>
        <tr>
            <th>True \\ Predicted</th>
            <th>🕳️ Pothole</th>
            <th>🔧 Damaged</th>
            <th>⚠️ Mixed</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="model-name">Pothole (503)</td>
            <td class="best-val">501</td>
            <td>0</td>
            <td>2</td>
        </tr>
        <tr>
            <td class="model-name">Damaged (102)</td>
            <td>3</td>
            <td class="best-val">99</td>
            <td>0</td>
        </tr>
        <tr>
            <td class="model-name">Mixed (28)</td>
            <td>2</td>
            <td>0</td>
            <td class="best-val">26</td>
        </tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)


# ── TAB 4: About ────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">About This Project</div>
        <p style="font-size:15px; color:#c8d6e5; line-height:1.7;">
            This application is part of a study titled <strong style="color:#ecf0f1;">Explainable Transfer
            Learning for Road Damage and Surface Defect Classification in Smart City
            Infrastructure Monitoring</strong>, developed at the University of Europe for
            Applied Sciences, Potsdam.
        </p>
        <p style="font-size:14px; color:#8e9eb3; line-height:1.7; margin-top:12px;">
            The system classifies road-surface images into three damage categories using
            an EfficientNetV2-S model fine-tuned on the Road Issues Detection Dataset.
            Grad-CAM and SHAP are used to verify that predictions are based on genuine
            defect regions rather than irrelevant background patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">Methodology Pipeline</div>
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-content">
                <div class="step-title">Dataset Preparation</div>
                <div class="step-desc">4,216 images filtered to 3 classes, stratified 70/15/15 split, weighted sampling for class imbalance.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-content">
                <div class="step-title">Preprocessing &amp; Augmentation</div>
                <div class="step-desc">Resize to 224x224, ImageNet normalisation, random crops, horizontal flips, brightness/contrast adjustment.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-content">
                <div class="step-title">Model Training</div>
                <div class="step-desc">Custom CNN, EfficientNetV2-S, and ConvNeXt-Tiny trained with AdamW, early stopping, and LR scheduling.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-content">
                <div class="step-title">Evaluation</div>
                <div class="step-desc">Accuracy, macro/weighted F1, top-2 accuracy, per-class metrics, confusion matrices, and efficiency analysis.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">5</div>
            <div class="step-content">
                <div class="step-title">Explainability</div>
                <div class="step-desc">Grad-CAM and SHAP verify that predictions rely on real road defects, not background shortcuts.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">6</div>
            <div class="step-content">
                <div class="step-title">Deployment</div>
                <div class="step-desc">Best model deployed as this Streamlit web app with live prediction and visual explanation.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">Tech Stack</div>
            <div style="font-size:14px; color:#c8d6e5; line-height:2;">
                🐍 Python &nbsp;·&nbsp; PyTorch<br>
                🖼️ EfficientNetV2-S (ImageNet pretrained)<br>
                🔍 Grad-CAM &nbsp;·&nbsp; SHAP<br>
                🚀 Streamlit &nbsp;·&nbsp; GitHub<br>
                📊 Kaggle Notebooks (training)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_l2:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">Links</div>
            <div style="font-size:14px; line-height:2.2;">
                <a href="https://github.com/AkshitRohilla-prog/road-issues-detection-ml-project" target="_blank" class="footer-link" style="font-size:14px;">📂 GitHub Repository</a><br>
                <a href="https://www.kaggle.com/code/akshitrohill/machine-learning/notebook" target="_blank" class="footer-link" style="font-size:14px;">📓 Kaggle Notebook</a><br>
                <a href="https://www.kaggle.com/datasets/programmerrdai/road-issues-detection-dataset" target="_blank" class="footer-link" style="font-size:14px;">📦 Dataset on Kaggle</a>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-left">
        Built by <strong>Akshit Rohilla</strong> &nbsp;·&nbsp;
        University of Europe for Applied Sciences, Potsdam
    </div>
    <div class="footer-right">
        <a href="mailto:akshitrohilla.in@gmail.com" class="footer-link">✉️ akshitrohilla.in@gmail.com</a>
        <a href="https://github.com/AkshitRohilla-prog/road-issues-detection-ml-project" target="_blank" class="footer-link">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)
