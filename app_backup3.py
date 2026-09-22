"""Human Emotion Detection From Facial Expressions - Streamlit UI.

The model, checkpoint loading, face detection, preprocessing and inference
functions are kept compatible with the supplied FER2013 project. This file
focuses on a premium presentation layer around that existing pipeline.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from model import DeepCNN
from transforms import GaussianDenoise, CLAHE
from preprocessing import apply_clahe, denoise_gaussian

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_EMOJI = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😊", "Sad": "😢", "Surprise": "😲", "Neutral": "😐",
}
EMOTION_ACCENT = {
    "Angry": "#ff5c5c", "Disgust": "#b9d66b", "Fear": "#c084fc",
    "Happy": "#f7c65b", "Sad": "#62a8ff", "Surprise": "#67e8f9", "Neutral": "#a7b0c0",
}
MODEL_INFO = {
    "Model": "Deep CNN",
    "Parameters": "~1.33M",
    "Dataset": "FER2013",
    "Image size": "48 × 48 grayscale",
    "Classes": "7",
    "Test accuracy": "65.19%",
}
PIPELINE_STEPS = [
    ("01", "Face", "Locate the largest visible face"),
    ("02", "Crop", "Isolate the facial region"),
    ("03", "Gray", "Convert to grayscale"),
    ("04", "Denoise", "Gaussian smoothing"),
    ("05", "Enhance", "CLAHE contrast enhancement"),
    ("06", "Resize", "Normalize to 48 × 48"),
    ("07", "Infer", "Deep CNN classification"),
]

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


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_model():
    device = get_device()
    if not CHECKPOINT_PATH.exists():
        st.error(
            f"Checkpoint not found at `{CHECKPOINT_PATH}`. This app only runs "
            "with the real trained Deep CNN checkpoint. Place `deep_cnn_best.pth` "
            "inside the `checkpoints` folder and restart the app."
        )
        st.stop()
    model = DeepCNN(num_classes=7)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, device


@torch.no_grad()
def predict(img_pil: Image.Image, model, device) -> np.ndarray:
    tensor = VAL_TFM(img_pil).unsqueeze(0).to(device)
    return torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()


def to_grayscale_48(img_pil: Image.Image) -> Image.Image:
    return img_pil.convert("L").resize((48, 48), Image.LANCZOS)


def detect_face(img_pil: Image.Image):
    gray = np.array(img_pil.convert("L"))
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
    return Image.fromarray(crop, mode="L"), (x, y, w, h)


def pipeline_preview_images(face_img: Image.Image):
    raw = np.array(face_img)
    denoised = denoise_gaussian(raw)
    enhanced = apply_clahe(denoised)
    return [
        ("Grayscale", Image.fromarray(raw)),
        ("Denoised", Image.fromarray(denoised)),
        ("CLAHE", Image.fromarray(enhanced)),
    ]


st.set_page_config(
    page_title="Human Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One style block only; all visible components are native Streamlit widgets.
st.markdown("""
<style>
:root {
    --bg: #080a0f;
    --panel: #10141c;
    --panel-2: #141925;
    --line: rgba(255,255,255,.09);
    --muted: #8d96a8;
    --text: #f4f6fb;
    --purple: #8b7cff;
    --cyan: #4fd1c5;
}
.stApp {
    background:
      radial-gradient(900px 450px at 7% -5%, rgba(139,124,255,.20), transparent 62%),
      radial-gradient(800px 420px at 98% 3%, rgba(79,209,197,.12), transparent 58%),
      var(--bg);
}
.block-container { max-width: 1220px; padding: 2.2rem 2.6rem 3rem; }
header[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0d1016; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding-top: 2rem; }

/* Native bordered containers become the design system's cards. */
div[class*="st-key-"] {
    background: linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.018)) !important;
    border: 1px solid var(--line) !important;
    border-radius: 22px !important;
    box-shadow: 0 16px 50px rgba(0,0,0,.20);
}
div[class*="st-key-hero-card"] { border-color: rgba(139,124,255,.22) !important; }
div[class*="st-key-result-card"] {
    background: linear-gradient(145deg, rgba(139,124,255,.12), rgba(79,209,197,.045)) !important;
    border-color: rgba(139,124,255,.32) !important;
}
div[class*="st-key-upload-card"] { background: rgba(16,20,28,.84) !important; }
div[class*="st-key-emotion-"] { min-height: 126px; }
div[class*="st-key-emotion-selected"] {
    border-color: rgba(139,124,255,.75) !important;
    box-shadow: 0 0 0 1px rgba(139,124,255,.18), 0 18px 45px rgba(85,72,190,.18);
}

/* Native typography. */
h1 { font-size: clamp(2.7rem, 5vw, 4.5rem) !important; line-height: .98 !important; letter-spacing: -0.045em !important; font-weight: 850 !important; }
h2, h3 { letter-spacing: -.025em; }
.block-container h1 {
    background: linear-gradient(90deg, #ffffff 8%, #a999ff 52%, #68ded2 92%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* Upload zone. */
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.025) !important;
    border: 1px dashed rgba(139,124,255,.42) !important;
    border-radius: 18px !important;
    padding: 1.4rem !important;
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--purple) !important; background: rgba(139,124,255,.045) !important; }

/* Progress bars. */
div[data-testid="stProgress"] div[role="progressbar"] { background: rgba(255,255,255,.07); border-radius: 99px; }
div[data-testid="stProgress"] div[role="progressbar"] > div { background: linear-gradient(90deg, var(--purple), var(--cyan)) !important; border-radius: 99px; }

/* Metrics and tabs. */
div[data-testid="stMetric"] { background: rgba(255,255,255,.025); border: 1px solid var(--line); border-radius: 16px; padding: .8rem 1rem; }
button[data-baseweb="tab"] { font-weight: 700; }
[data-testid="stImage"] img { border-radius: 16px; }

/* Make native expanders and alerts feel integrated. */
div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.018); }
div[data-testid="stAlert"] { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown("## 🧠 Emotion AI")
    st.caption("FACIAL EXPRESSION ANALYSIS")
    st.divider()
    st.markdown("**MODEL**")
    st.write("Deep CNN")
    st.caption("1.33M parameters · 7 classes")
    st.markdown("**DATASET**")
    st.write("FER2013")
    st.caption("48 × 48 grayscale faces")
    st.markdown("**RUNTIME**")
    st.write("Apple MPS / CUDA / CPU")
    st.divider()
    auto_detect = st.toggle("Auto-detect face", value=True)
    show_pipeline = st.toggle("Show preprocessing preview", value=False)
    st.divider()
    st.caption("FINAL-YEAR PROJECT")
    st.caption("Deep Learning · Computer Vision")

# --------------------------- HEADER ---------------------------
hero_left, hero_right = st.columns([4.6, 1.2], gap="large", vertical_alignment="center")
with hero_left:
    st.badge("AI  /  COMPUTER VISION", color="violet")
    st.markdown("# Human Emotion Detection")
    st.markdown("### Read facial expressions with deep learning.")
    st.write("Analyze a face image and estimate one of seven emotions using a Deep CNN trained on the FER2013 dataset.")
with hero_right:
    with st.container(border=True, key="hero-card"):
        st.markdown("# 🧠")
        st.caption("EMOTION AI")
        st.metric("Accuracy", "65.19%")

st.write("")

# --------------------------- MODEL LOAD ---------------------------
with st.spinner("Initializing emotion model…"):
    model, device = load_model()

# --------------------------- DETECTION WORKSPACE ---------------------------
st.markdown("## Analyze a face")
st.caption("Upload a clear, front-facing image or capture one with your camera.")

with st.container(border=True, key="upload-card"):
    upload_tab, camera_tab = st.tabs(["📁  Upload image", "📷  Capture photo"])
    img_pil = None
    cam_img = None
    with upload_tab:
        uploaded = st.file_uploader(
            "Drop an image here or browse your computer",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="visible",
        )
        if uploaded:
            try:
                img_pil = Image.open(uploaded)
                img_pil.load()
            except (UnidentifiedImageError, OSError):
                st.error("This file could not be read. Please choose a valid image.")
    with camera_tab:
        captured = st.camera_input("Take a photo", label_visibility="visible")
        if captured:
            try:
                cam_img = Image.open(captured)
                cam_img.load()
            except (UnidentifiedImageError, OSError):
                st.error("The captured image could not be read. Please try again.")

active_img = img_pil if img_pil is not None else cam_img

# --------------------------- RESULT ---------------------------
if active_img is not None:
    face_img = None
    bbox = None
    if auto_detect:
        face_img, bbox = detect_face(active_img)
    else:
        face_img = to_grayscale_48(active_img)

    if face_img is None:
        st.error("No face detected. Try a well-lit, front-facing photo with the face clearly visible.")
    else:
        probs = predict(face_img, model, device)
        pred_idx = int(probs.argmax())
        emotion = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])
        top3 = np.argsort(probs)[::-1][:3]

        st.write("")
        image_col, result_col = st.columns([1.05, 1.4], gap="large", vertical_alignment="center")
        with image_col:
            with st.container(border=True, key="preview-card"):
                st.caption("INPUT PREVIEW")
                st.image(active_img, use_container_width=True)
                if auto_detect:
                    st.success("Face detected")
                else:
                    st.info("Auto-detection disabled")
        with result_col:
            with st.container(border=True, key="result-card"):
                st.caption("PREDICTION")
                r1, r2 = st.columns([1, 2.7], vertical_alignment="center")
                with r1:
                    st.markdown(f"# {EMOTION_EMOJI[emotion]}")
                with r2:
                    st.markdown(f"## {emotion}")
                    st.badge("DETECTED EMOTION", color="violet")
                st.write("")
                st.progress(conf)
                st.metric("Model confidence", f"{conf * 100:.1f}%")
                if conf < 0.45:
                    st.warning("Low confidence. A clearer, front-facing image may improve the prediction.")

        st.write("")
        st.markdown("## Prediction breakdown")
        top_col, grid_col = st.columns([1.15, 2.1], gap="large")
        with top_col:
            with st.container(border=True, key="top3-card"):
                st.markdown("### Top 3")
                st.caption("Most probable emotions")
                for rank, idx in enumerate(top3, 1):
                    name = CLASS_NAMES[idx]
                    p = float(probs[idx])
                    st.write(f"**{rank:02d}  {EMOTION_EMOJI[name]} {name}**")
                    st.progress(p)
                    st.caption(f"{p * 100:.1f}%")
        with grid_col:
            st.markdown("### All emotions")
            cols = st.columns(4)
            for i, name in enumerate(CLASS_NAMES):
                key = "emotion-selected" if i == pred_idx else f"emotion-{i}"
                with cols[i % 4]:
                    with st.container(border=True, key=key):
                        st.markdown(f"### {EMOTION_EMOJI[name]}")
                        st.write(f"**{name}**")
                        st.caption(f"{probs[i] * 100:.1f}%")

        if show_pipeline:
            st.write("")
            st.markdown("## What the model sees")
            preview = pipeline_preview_images(face_img)
            pcols = st.columns(3)
            for col, (caption, im) in zip(pcols, preview):
                with col:
                    with st.container(border=True, key=f"preview-{caption}"):
                        st.image(im, use_container_width=True)
                        st.caption(caption)

# --------------------------- EXPLAINER ---------------------------
st.write("")
st.markdown("## How it works")
pipe_cols = st.columns(len(PIPELINE_STEPS))
for col, (num, title, desc) in zip(pipe_cols, PIPELINE_STEPS):
    with col:
        with st.container(border=True, key=f"pipe-{num}"):
            st.caption(num)
            st.markdown(f"**{title}**")
            st.caption(desc)

st.write("")
model_left, model_right = st.columns([1.3, 1], gap="large")
with model_left:
    with st.container(border=True, key="model-card"):
        st.caption("MODEL PROFILE")
        st.markdown("## Deep CNN")
        st.write("A custom convolutional network with attention blocks, global average pooling and fully connected classification layers.")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Parameters", "1.33M")
        with m2: st.metric("Classes", "7")
        with m3: st.metric("Test accuracy", "65.19%")
with model_right:
    with st.container(border=True, key="dataset-card"):
        st.caption("DATASET")
        st.markdown("## FER2013")
        st.write("Facial expression images used to train and evaluate the seven-class emotion classifier.")
        st.write("**Input:** 48 × 48 grayscale")
        st.write("**Classes:** Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral")

with st.expander("📘  About this project"):
    st.write(
        "Human facial expressions provide useful visual cues about emotional states. "
        "This project combines OpenCV face detection and image enhancement with a custom "
        "Deep CNN to classify facial expressions into seven FER2013 emotion categories."
    )

st.divider()
footer_a, footer_b = st.columns([3, 1])
with footer_a:
    st.caption("HUMAN EMOTION DETECTION  ·  DEEP LEARNING  ·  FER2013")
with footer_b:
    st.caption(f"Runtime: {str(device).upper()}")
