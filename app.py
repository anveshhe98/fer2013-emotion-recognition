"""Human Emotion Detection From Facial Expressions - premium product UI."""
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
    "Angry": "#ff5f57", "Disgust": "#a8c76b", "Fear": "#b99cff",
    "Happy": "#ffd166", "Sad": "#63a4ff", "Surprise": "#5eead4", "Neutral": "#a8b0bd",
}
PIPELINE_STEPS = [
    ("01", "FACE", "Locate the visible face"),
    ("02", "CROP", "Isolate facial region"),
    ("03", "GRAY", "Convert to grayscale"),
    ("04", "DENOISE", "Gaussian smoothing"),
    ("05", "ENHANCE", "CLAHE contrast"),
    ("06", "RESIZE", "Normalize to 48 × 48"),
    ("07", "INFER", "Deep CNN classification"),
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
        ("GRAYSCALE", Image.fromarray(raw)),
        ("DENOISED", Image.fromarray(denoised)),
        ("CLAHE", Image.fromarray(enhanced)),
    ]


st.set_page_config(
    page_title="Emotion AI — Facial Expression Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ============================================================
   PREMIUM PRODUCT SYSTEM — deliberately not a dashboard style
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --black:#050607;
  --ink:#f5f5f5;
  --muted:#8c8f96;
  --muted2:#5f636b;
  --line:rgba(255,255,255,.11);
  --line2:rgba(255,255,255,.06);
  --violet:#8c7bff;
  --cyan:#63e6d5;
}

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
  background:
    radial-gradient(800px 500px at 82% -8%, rgba(98,82,255,.16), transparent 65%),
    radial-gradient(700px 450px at 8% 20%, rgba(0,190,180,.07), transparent 70%),
    #050607;
  color:var(--ink);
}
.block-container { max-width:1440px; padding:0 54px 80px; }
header[data-testid="stHeader"] { background:rgba(5,6,7,.72); backdrop-filter:blur(18px); }
footer, #MainMenu { visibility:hidden; }

/* Hide Streamlit chrome that makes the page feel like a notebook. */
[data-testid="stToolbar"] { display:none; }

/* Sidebar becomes a restrained utility drawer. */
section[data-testid="stSidebar"] {
  background:#090a0c; border-right:1px solid var(--line); width:300px !important;
}
section[data-testid="stSidebar"] > div { padding:28px 24px; }

/* Native typography. */
h1,h2,h3,p { font-family:'DM Sans',sans-serif !important; }
h1 { font-size:clamp(4.4rem,8.5vw,8.8rem) !important; line-height:.87 !important; letter-spacing:-.075em !important; font-weight:700 !important; margin:0 !important; }
h2 { font-size:clamp(2.2rem,4vw,4.2rem) !important; line-height:.98 !important; letter-spacing:-.055em !important; font-weight:600 !important; }
h3 { letter-spacing:-.035em !important; }
[data-testid="stCaptionContainer"] { color:var(--muted) !important; }

/* Top product bar. */
.product-bar {
  height:74px; display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid var(--line2); margin:0 -54px 0; padding:0 54px;
  letter-spacing:.04em; font-size:12px; text-transform:uppercase; color:#c6c8cd;
}
.brand { font-weight:700; color:#fff; letter-spacing:.13em; }
.nav-meta { display:flex; gap:28px; color:#737780; }
.status-dot { color:#68e0c5; margin-right:7px; text-shadow:0 0 14px #68e0c5; }

/* Hero. */
.hero { padding:90px 0 120px; min-height:590px; position:relative; }
.eyebrow { color:#a89dff; letter-spacing:.24em; font-size:11px; font-weight:700; text-transform:uppercase; margin-bottom:28px; }
.hero h1 { max-width:1000px; background:linear-gradient(100deg,#fff 10%,#d8d1ff 48%,#83ded2 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-copy { max-width:650px; margin-top:36px; font-size:18px; line-height:1.65; color:#a6a9b1; }
.hero-line { width:100%; height:1px; background:linear-gradient(90deg,rgba(255,255,255,.35),rgba(255,255,255,0)); margin-top:68px; }
.hero-stat { position:absolute; right:0; bottom:122px; text-align:right; }
.hero-stat .label { font-size:10px; letter-spacing:.2em; color:#686d76; text-transform:uppercase; }
.hero-stat .value { font-family:'Space Grotesk',sans-serif; font-size:56px; letter-spacing:-.06em; font-weight:600; }

/* Section framing. */
.section { padding:110px 0; border-top:1px solid var(--line2); }
.section-index { font-size:10px; color:#777b83; letter-spacing:.25em; font-weight:700; margin-bottom:22px; }
.section-title { font-size:clamp(2.6rem,5vw,5rem); letter-spacing:-.065em; line-height:.95; font-weight:600; }
.section-copy { color:#858991; max-width:620px; line-height:1.65; font-size:16px; margin-top:18px; }

/* Native containers: subtle, not card-heavy. */
div[class*="st-key-workspace"], div[class*="st-key-result"], div[class*="st-key-profile"] {
  background:rgba(255,255,255,.018) !important;
  border:1px solid var(--line) !important;
  border-radius:4px !important;
  box-shadow:none !important;
}
div[class*="st-key-preview"] { background:rgba(255,255,255,.012) !important; border:1px solid var(--line) !important; border-radius:4px !important; }

/* Upload control — make it feel like a product input, not a widget. */
div[data-testid="stFileUploaderDropzone"] {
  background:transparent !important; border:1px dashed rgba(255,255,255,.19) !important;
  border-radius:2px !important; min-height:155px; padding:24px !important;
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color:#8c7bff !important; background:rgba(140,123,255,.035) !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] { color:#9a9da5 !important; }

/* Tabs. */
button[data-baseweb="tab"] { background:transparent !important; border:0 !important; color:#777b83 !important; font-weight:600 !important; letter-spacing:.03em; }
button[data-baseweb="tab"][aria-selected="true"] { color:#fff !important; }

/* Images. */
[data-testid="stImage"] img { border-radius:2px !important; border:1px solid rgba(255,255,255,.08); }

/* Confidence bar. */
div[data-testid="stProgress"] { margin:8px 0 4px; }
div[data-testid="stProgress"] > div { background:rgba(255,255,255,.07) !important; border-radius:0 !important; height:4px !important; }
div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#8c7bff,#62dfd0) !important; border-radius:0 !important; }

/* Metrics stripped down. */
div[data-testid="stMetric"] { background:transparent !important; border:0 !important; padding:0 !important; }
div[data-testid="stMetricLabel"] { color:#666a73 !important; text-transform:uppercase; letter-spacing:.14em; font-size:10px !important; }
div[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif !important; font-size:32px !important; letter-spacing:-.04em; }

/* Expanders and alerts. */
div[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:2px !important; background:transparent !important; }
div[data-testid="stAlert"] { border-radius:2px !important; }

/* Buttons. */
.stButton > button { border-radius:2px !important; background:#fff !important; color:#08090a !important; border:1px solid #fff !important; font-weight:700 !important; padding:.65rem 1.2rem !important; }

/* Pipeline. */
.pipeline-item { border-top:1px solid var(--line); padding:25px 4px 12px; min-height:145px; }
.pipeline-no { color:#60646c; font:500 11px 'Space Grotesk',sans-serif; letter-spacing:.12em; }
.pipeline-name { font-size:18px; font-weight:600; margin-top:28px; letter-spacing:-.02em; }
.pipeline-desc { color:#70747c; font-size:12px; line-height:1.45; margin-top:8px; }

/* Emotion rows. */
.emotion-row { display:flex; align-items:center; gap:18px; border-top:1px solid var(--line2); padding:17px 0; }
.emotion-name { width:105px; font-weight:600; font-size:14px; }
.emotion-track { flex:1; height:3px; background:#202329; }
.emotion-fill { height:100%; background:linear-gradient(90deg,#8c7bff,#62dfd0); }
.emotion-value { width:58px; text-align:right; color:#a3a6ad; font:600 13px 'Space Grotesk',sans-serif; }

/* Result typography. */
.result-kicker { color:#777b84; font-size:10px; letter-spacing:.24em; text-transform:uppercase; }
.result-emotion { font-size:clamp(4rem,7vw,7.5rem); font-weight:600; letter-spacing:-.08em; line-height:.85; margin-top:15px; }
.result-sub { color:#777b83; font-size:13px; letter-spacing:.08em; text-transform:uppercase; margin-top:22px; }
.scan-label { color:#6f7480; font-size:9px; letter-spacing:.2em; text-transform:uppercase; margin-bottom:12px; }

/* Technical specification strip. */
.spec { border-top:1px solid var(--line); padding:24px 0; }
.spec-label { color:#62666f; text-transform:uppercase; letter-spacing:.18em; font-size:9px; }
.spec-value { font:500 22px 'Space Grotesk',sans-serif; margin-top:8px; letter-spacing:-.025em; }

/* Footer. */
.footer { border-top:1px solid var(--line); padding:28px 0; color:#5d6169; font-size:10px; letter-spacing:.16em; text-transform:uppercase; }

@media (max-width: 900px) {
  .block-container { padding:0 22px 60px; }
  .product-bar { margin:0 -22px; padding:0 22px; }
  .nav-meta { display:none; }
  .hero { padding-top:65px; min-height:auto; }
  .hero-stat { position:static; text-align:left; margin-top:55px; }
  .section { padding:75px 0; }
}
</style>
""", unsafe_allow_html=True)

# Product navigation
st.markdown("""
<div class="product-bar">
  <div class="brand">EMOTION AI</div>
  <div class="nav-meta">
    <span>FACIAL INTELLIGENCE</span>
    <span>FER2013</span>
    <span><span class="status-dot">●</span>SYSTEM READY</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Minimal utility sidebar
with st.sidebar:
    st.markdown("### 🧠 EMOTION AI")
    st.caption("FACIAL EXPRESSION INTELLIGENCE")
    st.divider()
    st.caption("MODEL")
    st.write("Deep CNN · ~1.33M parameters")
    st.caption("DATASET")
    st.write("FER2013 · 7 emotions")
    st.caption("RUNTIME")
    st.write("Apple MPS / CUDA / CPU")
    st.divider()
    auto_detect = st.toggle("Auto-detect face", value=True)
    show_pipeline = st.toggle("Show preprocessing preview", value=False)

# Hero
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI / COMPUTER VISION / FER2013</div>
  <h1>Human<br>Emotion<br>Detection</h1>
  <div class="hero-copy">Read facial expressions with deep learning. Transform a face into an interpretable seven-class emotion signal using a trained Deep CNN.</div>
  <div class="hero-line"></div>
  <div class="hero-stat"><div class="label">Model accuracy</div><div class="value">65.19%</div></div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Initializing emotion model…"):
    model, device = load_model()

# Analysis section
st.markdown("""
<div class="section">
  <div class="section-index">01 / LIVE ANALYSIS</div>
  <div class="section-title">See what the model sees.</div>
  <div class="section-copy">Upload a clear, front-facing image or capture one with your camera. The trained network performs the actual classification.</div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True, key="workspace"):
    upload_tab, camera_tab = st.tabs(["UPLOAD IMAGE", "CAPTURE PHOTO"])
    img_pil = None
    cam_img = None
    with upload_tab:
        uploaded = st.file_uploader("Drop an image here or browse your computer", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="visible")
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

        st.markdown('<div class="section" style="padding-bottom:38px"><div class="section-index">02 / RESULT</div><div class="section-title">From pixels to perception.</div></div>', unsafe_allow_html=True)

        left, right = st.columns([1.08, 1.35], gap="large", vertical_alignment="top")
        with left:
            with st.container(border=True, key="preview"):
                st.markdown('<div class="scan-label">SOURCE IMAGE</div>', unsafe_allow_html=True)
                st.image(active_img, use_container_width=True)
                if auto_detect:
                    st.success("FACE DETECTED")
                else:
                    st.info("AUTO-DETECTION DISABLED")
        with right:
            with st.container(border=True, key="result"):
                st.markdown('<div class="result-kicker">DETECTED EMOTION</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-emotion">{EMOTION_EMOJI[emotion]} {emotion}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-sub">Confidence · {conf*100:.1f}%</div>', unsafe_allow_html=True)
                st.progress(conf)
                st.write("")
                r1, r2, r3 = st.columns(3)
                with r1: st.metric("Emotion", emotion)
                with r2: st.metric("Confidence", f"{conf*100:.1f}%")
                with r3: st.metric("Classes", "7")
                if conf < 0.45:
                    st.warning("Low confidence. A clearer, front-facing image may improve the prediction.")

        st.write("")
        st.markdown('<div class="section-index">03 / EMOTION PROFILE</div>', unsafe_allow_html=True)
        prof_left, prof_right = st.columns([1.25, 1], gap="large")
        with prof_left:
            st.markdown("### Probability profile")
            for i in np.argsort(probs)[::-1]:
                name = CLASS_NAMES[i]
                pct = float(probs[i]) * 100
                accent = EMOTION_ACCENT[name]
                st.markdown(f'''<div class="emotion-row"><div class="emotion-name">{EMOTION_EMOJI[name]} {name}</div><div class="emotion-track"><div class="emotion-fill" style="width:{max(pct,1):.2f}%;background:{accent};"></div></div><div class="emotion-value">{pct:.1f}%</div></div>''', unsafe_allow_html=True)
        with prof_right:
            st.markdown("### Top 3 signal")
            for rank, idx in enumerate(top3, 1):
                name = CLASS_NAMES[idx]
                pct = float(probs[idx])
                st.markdown(f"**0{rank}  {EMOTION_EMOJI[name]} {name}**")
                st.progress(pct)
                st.caption(f"{pct*100:.1f}% probability")

        if show_pipeline:
            st.markdown('<div class="section-index" style="margin-top:65px">04 / WHAT THE MODEL SEES</div>', unsafe_allow_html=True)
            preview = pipeline_preview_images(face_img)
            pcols = st.columns(3)
            for col, (caption, im) in zip(pcols, preview):
                with col:
                    st.image(im, use_container_width=True)
                    st.caption(caption)

# Method section
st.markdown("""
<div class="section">
  <div class="section-index">05 / SYSTEM</div>
  <div class="section-title">A signal, not a guess.</div>
  <div class="section-copy">The image passes through a compact computer-vision pipeline before the Deep CNN produces a probability distribution across seven facial-expression classes.</div>
</div>
""", unsafe_allow_html=True)

pipe_cols = st.columns(7, gap="small")
for col, (num, title, desc) in zip(pipe_cols, PIPELINE_STEPS):
    with col:
        st.markdown(f'<div class="pipeline-item"><div class="pipeline-no">{num}</div><div class="pipeline-name">{title}</div><div class="pipeline-desc">{desc}</div></div>', unsafe_allow_html=True)

st.write("")

# Technical profile — deliberately flat, like a product spec sheet
st.markdown('<div class="section-index" style="margin-top:45px">06 / TECHNICAL PROFILE</div>', unsafe_allow_html=True)
spec_cols = st.columns(3, gap="large")
specs = [
    ("MODEL", "Deep CNN"),
    ("PARAMETERS", "~1.33M"),
    ("DATASET", "FER2013"),
    ("INPUT", "48 × 48 grayscale"),
    ("CLASSES", "7 emotions"),
    ("TEST ACCURACY", "65.19%"),
]
for i, (label, value) in enumerate(specs):
    with spec_cols[i % 3]:
        st.markdown(f'<div class="spec"><div class="spec-label">{label}</div><div class="spec-value">{value}</div></div>', unsafe_allow_html=True)

st.write("")
with st.expander("PROJECT / ABOUT"):
    st.write("Human facial expressions provide useful visual cues about emotional states. This project combines OpenCV face detection and image enhancement with a custom Deep CNN to classify facial expressions into seven FER2013 emotion categories.")

st.markdown(f'<div class="footer"><span>EMOTION AI · DEEP LEARNING · FER2013</span><span style="float:right">RUNTIME · {str(device).upper()}</span></div>', unsafe_allow_html=True)
