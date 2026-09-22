"""
HUMAN EMOTION DETECTION FROM FACIAL EXPRESSIONS
Streamlit dashboard — FER2013 Facial Emotion Recognition
INSAT GL4 · Image Processing Project · 2026

This file is a UI/UX redesign of the original working app.py.
The model architecture, checkpoint file, and inference pipeline are
UNCHANGED — only presentation, layout and messaging were rebuilt.

Run:
    streamlit run app.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from model import DeepCNN
from transforms import GaussianDenoise, CLAHE
from preprocessing import apply_clahe, denoise_gaussian

# =============================================================================
# CONSTANTS  (unchanged from the original working app — do not edit values
# here without retraining, they must match the checkpoint's training config)
# =============================================================================

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_EMOJI = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😊", "Sad": "😢", "Surprise": "😲", "Neutral": "😐",
}
EMOTION_COLOR = {
    "Angry": "#e74c3c", "Disgust": "#e67e22", "Fear": "#f39c12",
    "Happy": "#2ecc71", "Sad": "#3498db", "Surprise": "#9b59b6",
    "Neutral": "#95a5a6",
}

# Static model card info (matches the trained deep_cnn_best.pth checkpoint —
# see notebooks/cr4_training.ipynb and results/cr4_summary.png for the source
# of these numbers). Purely descriptive — does not affect inference.
MODEL_INFO = {
    "Model":          "Deep CNN",
    "Parameters":     "~1.33M",
    "Dataset":        "FER2013",
    "Image size":     "48 × 48 grayscale",
    "Classes":        "7",
    "Test accuracy":  "65.19%",
}

# Exact same validation-time transform pipeline used during training/eval.
VAL_TFM = transforms.Compose([
    GaussianDenoise(),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.563], std=[0.2627]),
])

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CHECKPOINT_PATH = ROOT / "checkpoints" / "deep_cnn_best.pth"

# =============================================================================
# DEVICE SELECTION — MPS (Apple Silicon) > CUDA > CPU, no CUDA-only deps
# =============================================================================

def get_device() -> torch.device:
    """Pick the best available device without requiring any CUDA-only package.

    Order of preference:
      1. Apple Silicon GPU (MPS) — for Macs
      2. NVIDIA GPU (CUDA) — if present
      3. CPU — universal fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# =============================================================================
# MODEL LOADING — unchanged logic, cached so it only loads once per session.
# There is NO fallback / mock model here: if the real checkpoint is missing
# or fails to load, the app stops with a clear error instead of guessing.
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    device = get_device()

    if not CHECKPOINT_PATH.exists():
        # Hard stop — never substitute a random/untrained model.
        st.error(
            "❌ **Checkpoint not found.**\n\n"
            f"Expected the trained weights at:\n`{CHECKPOINT_PATH}`\n\n"
            "This app only runs with the real trained Deep CNN checkpoint. "
            "Please place `deep_cnn_best.pth` in the `checkpoints/` folder "
            "and restart the app."
        )
        st.stop()

    model = DeepCNN(num_classes=7)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, device

# =============================================================================
# INFERENCE — unchanged from the original app
# =============================================================================

@torch.no_grad()
def predict(img_pil: Image.Image, model, device) -> np.ndarray:
    tensor = VAL_TFM(img_pil).unsqueeze(0).to(device)
    return torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

# =============================================================================
# IMAGE PREPARATION — unchanged from the original app
# =============================================================================

def to_grayscale_48(img_pil: Image.Image) -> Image.Image:
    return img_pil.convert("L").resize((48, 48), Image.LANCZOS)


def detect_face(img_pil: Image.Image):
    """Return (face PIL 48x48, bbox) or (None, None) if no face found."""
    gray = np.array(img_pil.convert("L"))
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
    crop = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
    return Image.fromarray(crop, mode="L"), (x, y, w, h)

# =============================================================================
# VISUALIZATIONS — restyled for the dark dashboard theme
# =============================================================================

def _prob_chart(probs: np.ndarray) -> plt.Figure:
    """Horizontal bar chart of all 7 class probabilities, dark-theme styled."""
    pred_idx = int(probs.argmax())
    colors = [
        EMOTION_COLOR[c] if i == pred_idx else "#3a3f4b"
        for i, c in enumerate(CLASS_NAMES)
    ]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor="none")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=9, color="#c9ccd1")
    ax.tick_params(colors="#c9ccd1", labelsize=9)
    ax.axvline(50, color="#4a4f5b", linestyle="--", alpha=0.7, linewidth=1)

    for bar, p in zip(bars, probs):
        ax.text(p * 100 + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{p*100:.1f}%", va="center", fontsize=8.5, color="#e6e6e6")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


def _pipeline_chart(img_pil: Image.Image) -> plt.Figure:
    """Visualize each preprocessing step on the current image (dark theme)."""
    raw = np.array(img_pil.convert("L").resize((48, 48), Image.LANCZOS))
    denoised = denoise_gaussian(raw)
    enhanced = apply_clahe(denoised)
    normalized = (enhanced / 255.0 - 0.563) / 0.2627

    steps = [
        ("1. Raw (48×48)", raw, "gray", None),
        ("2. Gaussian denoising", denoised, "gray", None),
        ("3. CLAHE", enhanced, "gray", None),
        ("4. Normalized", normalized, "RdBu", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    fig.patch.set_alpha(0.0)
    for ax, (label, data, cmap, symmetric) in zip(axes, steps):
        kwargs = dict(cmap=cmap)
        if symmetric:
            vmax = max(abs(data.min()), abs(data.max()))
            kwargs.update(vmin=-vmax, vmax=vmax)
        ax.imshow(data, **kwargs)
        ax.set_title(label, fontsize=8.5, fontweight="bold", color="#e6e6e6")
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig

# =============================================================================
# SMALL UI HELPERS
# =============================================================================

def metric_card(label: str, value: str) -> str:
    """Return HTML for one compact stat card used in the model info grid."""
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """


def top3_row(rank: int, name: str, prob: float) -> str:
    color = EMOTION_COLOR[name]
    emoji = EMOTION_EMOJI[name]
    pct = prob * 100
    return f"""
    <div class="top3-row">
        <div class="top3-rank">#{rank}</div>
        <div class="top3-label">{emoji} {name}</div>
        <div class="top3-bar-bg">
            <div class="top3-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
        </div>
        <div class="top3-pct">{pct:.1f}%</div>
    </div>
    """

# =============================================================================
# PREDICTION + RESULT RENDERING
# =============================================================================

def render_result(img_pil: Image.Image, model, device, auto_detect: bool, show_pipeline: bool):
    """Run face detection + inference on img_pil and render the result section."""

    if auto_detect:
        face_img, bbox = detect_face(img_pil)
        if face_img is None:
            st.error("🚫 Please upload an image with a clearly visible face.")
            return
        st.success("✅ Face detected successfully")
    else:
        face_img = to_grayscale_48(img_pil)

    probs = predict(face_img, model, device)
    pred_idx = int(probs.argmax())
    emotion = CLASS_NAMES[pred_idx]
    conf = probs[pred_idx]
    top3 = np.argsort(probs)[::-1][:3]

    st.markdown("<div class='section-title'>Detection Result</div>", unsafe_allow_html=True)

    col_face, col_result = st.columns([1, 2], gap="large")

    with col_face:
        st.image(face_img, caption="Model input (48×48 grayscale)", width=200)

    with col_result:
        color = EMOTION_COLOR[emotion]
        emoji = EMOTION_EMOJI[emotion]
        st.markdown(
            f"""
            <div class="result-banner" style="border-color:{color}33; background:{color}14;">
                <div class="result-emoji">{emoji}</div>
                <div>
                    <div class="result-emotion" style="color:{color};">{emotion}</div>
                    <div class="result-confidence">Confidence: <b>{conf*100:.1f}%</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if conf < 0.45:
            st.warning("⚠️ Low confidence — try a clearer, well-lit, front-facing photo.")

        st.markdown("<div class='top3-title'>Top 3 Predictions</div>", unsafe_allow_html=True)
        rows_html = "".join(
            top3_row(rank, CLASS_NAMES[i], probs[i]) for rank, i in enumerate(top3, start=1)
        )
        st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Emotion Probability Distribution</div>", unsafe_allow_html=True)
    st.pyplot(_prob_chart(probs), use_container_width=True)

    if show_pipeline:
        st.markdown("<div class='section-title'>Preprocessing Pipeline (this image)</div>", unsafe_allow_html=True)
        st.pyplot(_pipeline_chart(img_pil), use_container_width=True)

# =============================================================================
# PAGE CONFIG + GLOBAL CSS
# =============================================================================

st.set_page_config(
    page_title="Human Emotion Detection From Facial Expressions",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    /* ---- Global type & spacing ---- */
    html, body, [class*="css"] { font-family: 'Segoe UI', -apple-system, sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    /* ---- Hero header ---- */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #8f87ff, #6C63FF 40%, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #aab0bb;
        max-width: 720px;
        margin-bottom: 1.6rem;
    }

    /* ---- Section titles ---- */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e6e6e6;
        margin: 1.6rem 0 0.7rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #2a2f3a;
    }

    /* ---- Stat / model-info cards ---- */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 0.5rem;
    }
    .stat-card {
        background: #161B22;
        border: 1px solid #2a2f3a;
        border-radius: 14px;
        padding: 14px 16px;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #8b93a1;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f0f0f0;
    }

    /* ---- Result banner ---- */
    .result-banner {
        display: flex;
        align-items: center;
        gap: 18px;
        border: 1px solid;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .result-emoji { font-size: 3rem; line-height: 1; }
    .result-emotion { font-size: 1.9rem; font-weight: 800; }
    .result-confidence { color: #cfd3da; font-size: 0.95rem; margin-top: 2px; }

    /* ---- Top-3 list ---- */
    .top3-title { font-weight: 700; color: #e6e6e6; margin: 6px 0 8px 0; font-size: 0.95rem; }
    .top3-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .top3-rank { color: #8b93a1; font-size: 0.8rem; width: 20px; }
    .top3-label { width: 130px; font-size: 0.9rem; color: #e6e6e6; }
    .top3-bar-bg { flex: 1; background: #262b36; border-radius: 6px; height: 10px; overflow: hidden; }
    .top3-bar-fill { height: 100%; border-radius: 6px; }
    .top3-pct { width: 50px; text-align: right; font-size: 0.85rem; color: #cfd3da; }

    /* ---- Upload card ---- */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 1.5px dashed #3a3f4b;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HOME / HEADER
# =============================================================================

st.markdown('<div class="hero-title">🧠 Human Emotion Detection From Facial Expressions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">A deep learning system that analyzes a facial image and classifies '
    'it into one of seven emotions — Angry, Disgust, Fear, Happy, Sad, Surprise or Neutral — '
    'using a custom Deep CNN trained on the FER2013 dataset.</div>',
    unsafe_allow_html=True,
)

# Load the model once (cached). Stops the app with a clear error if the
# checkpoint is missing — see load_model() above.
with st.spinner("Loading Deep CNN model…"):
    model, device = load_model()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### 🧠 Emotion Detection")
    st.caption("Final Year College Project · FER2013")

    st.divider()
    st.markdown("#### ⚙️ Settings")
    auto_detect = st.toggle("Auto-detect face", value=True,
                             help="Uses OpenCV Haar Cascade to find and crop the face automatically.")
    show_pipeline = st.toggle("Show preprocessing steps", value=False,
                               help="Visualize denoising, CLAHE and normalization on the uploaded image.")

    st.divider()
    st.markdown("#### 📊 Model")
    st.markdown(
        f"- **Architecture:** {MODEL_INFO['Model']}\n"
        f"- **Parameters:** {MODEL_INFO['Parameters']}\n"
        f"- **Test accuracy:** {MODEL_INFO['Test accuracy']}\n"
    )

    st.markdown("#### 🗂️ Dataset")
    st.markdown(
        f"- **Source:** {MODEL_INFO['Dataset']}\n"
        f"- **Input size:** {MODEL_INFO['Image size']}\n"
        f"- **Classes:** {MODEL_INFO['Classes']} emotions\n"
    )

    st.divider()
    st.markdown("#### 🖥️ Device")
    st.code(str(device), language=None)

    st.divider()
    with st.expander("ℹ️ About"):
        st.markdown(
            "Human facial expressions provide important information about "
            "emotional states. This project uses deep learning to classify "
            "facial expressions into seven emotion categories using the "
            "FER2013 dataset."
        )

# =============================================================================
# MODEL INFORMATION CARD (main page)
# =============================================================================

st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="stat-grid">'
    + metric_card("Model", MODEL_INFO["Model"])
    + metric_card("Parameters", MODEL_INFO["Parameters"])
    + metric_card("Dataset", MODEL_INFO["Dataset"])
    + metric_card("Image size", MODEL_INFO["Image size"])
    + metric_card("Classes", MODEL_INFO["Classes"])
    + metric_card("Test accuracy", MODEL_INFO["Test accuracy"])
    + '</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# UPLOAD / CAPTURE
# =============================================================================

st.markdown('<div class="section-title">Upload a Face Image</div>', unsafe_allow_html=True)

tab_upload, tab_webcam = st.tabs(["📁 Upload Image", "📷 Capture Photo"])

with tab_upload:
    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG, BMP, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded:
        try:
            img_pil = Image.open(uploaded)
            img_pil.load()  # force decode now so corrupt files raise here
        except (UnidentifiedImageError, OSError):
            st.error("🚫 This file could not be read as an image. Please upload a valid JPG, PNG, BMP or WEBP file.")
        else:
            st.image(img_pil, caption="Uploaded image", width=280)
            render_result(img_pil, model, device, auto_detect, show_pipeline)

with tab_webcam:
    st.caption("Click **Take photo**, then wait for the prediction to appear below.")
    captured = st.camera_input("Take photo")
    if captured:
        try:
            img_pil = Image.open(captured)
            img_pil.load()
        except (UnidentifiedImageError, OSError):
            st.error("🚫 Could not read the captured photo. Please try again.")
        else:
            render_result(img_pil, model, device, auto_detect, show_pipeline)

# =============================================================================
# PREPROCESSING PIPELINE (static explanation, always visible/expandable)
# =============================================================================

st.markdown('<div class="section-title">Pipeline &amp; Project Details</div>', unsafe_allow_html=True)

with st.expander("🔧 Preprocessing Pipeline"):
    st.markdown(
        "1. **Face detection** — OpenCV Haar Cascade locates the face in the uploaded image.\n"
        "2. **Face cropping** — the largest detected face is cropped from the original image.\n"
        "3. **Grayscale conversion** — the crop is converted to single-channel grayscale.\n"
        "4. **Gaussian denoising** — a 3×3 Gaussian blur (σ = 0.8) removes sensor/compression noise.\n"
        "5. **CLAHE enhancement** — Contrast Limited Adaptive Histogram Equalization "
        "(clip limit 2.0, tile size 4×4) improves local contrast.\n"
        "6. **Resize to 48×48** — matches the exact input resolution the model was trained on.\n"
        "7. **Normalization** — pixel values are scaled and normalized (mean = 0.563, std = 0.2627) "
        "before being fed to the network."
    )

with st.expander("📘 About This Project"):
    st.markdown(
        "Human facial expressions provide important information about emotional states. "
        "This project uses deep learning to classify facial expressions into seven emotion "
        "categories using the FER2013 dataset.\n\n"
        "The pipeline combines classical computer vision (OpenCV Haar Cascade face detection "
        "and image preprocessing) with a custom Deep CNN — using double-convolution blocks and "
        "Squeeze-and-Excitation attention — trained from scratch on FER2013 to recognize "
        "**Angry, Disgust, Fear, Happy, Sad, Surprise** and **Neutral** expressions."
    )
