"""Human Emotion Detection — premium Streamlit presentation UI.

The FER2013 Deep CNN inference pipeline is kept compatible with the supplied
project. The redesign focuses on the presentation layer and does not create
mock or fallback predictions.
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
    "Angry": "#ff647c", "Disgust": "#b8d66f", "Fear": "#c58aff",
    "Happy": "#ffd166", "Sad": "#63a4ff", "Surprise": "#5de4e6", "Neutral": "#a8b1c2",
}
PIPELINE_STEPS = [
    ("01", "FACE", "Locate the dominant face"),
    ("02", "CROP", "Isolate the facial region"),
    ("03", "GRAY", "Convert to grayscale"),
    ("04", "DENOISE", "Gaussian smoothing"),
    ("05", "CLAHE", "Enhance local contrast"),
    ("06", "48×48", "Normalize model input"),
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
        ("Grayscale", Image.fromarray(raw)),
        ("Denoised", Image.fromarray(denoised)),
        ("CLAHE", Image.fromarray(enhanced)),
    ]


st.set_page_config(
    page_title="Emotion AI — Facial Expression Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Decorative HTML is deliberately isolated here and always rendered with
# unsafe_allow_html=True. Functional controls remain native Streamlit widgets.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg:#06070b; --surface:#0c0f15; --surface2:#11151d; --line:rgba(255,255,255,.085);
  --text:#f7f7fb; --muted:#858da0; --violet:#8b7cff; --cyan:#58e0d0;
}
.stApp { background:
  radial-gradient(800px 520px at 78% -12%, rgba(93,75,255,.20), transparent 65%),
  radial-gradient(700px 480px at 10% 12%, rgba(0,210,190,.08), transparent 65%),
  linear-gradient(180deg,#06070b 0%,#080a0f 100%); color:var(--text); }
.block-container { max-width:1440px; padding:1.4rem 2.8rem 3.5rem; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk','DM Sans',sans-serif !important; }
h1 { font-size:clamp(3.2rem,6vw,6.5rem)!important; line-height:.94!important; letter-spacing:-.055em!important; font-weight:700!important; }
h2 { font-size:2rem!important; letter-spacing:-.04em!important; }
h3 { letter-spacing:-.025em!important; }
.block-container h1 { background:linear-gradient(100deg,#fff 15%,#b7a8ff 52%,#65e4d7 92%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
header[data-testid="stHeader"] { background:transparent; }
footer,#MainMenu { visibility:hidden; }
section[data-testid="stSidebar"] { background:rgba(8,10,15,.96); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding:1.6rem 1.1rem; }

/* Decorative cards used around real Streamlit controls. */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:24px!important; border:1px solid var(--line)!important; background:linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.012))!important; box-shadow:0 24px 70px rgba(0,0,0,.24); }
div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:24px!important; }

/* Hero / status cards. */
.hero-orb { width:92px;height:92px;border-radius:28px;display:grid;place-items:center;font-size:44px;
 background:radial-gradient(circle at 35% 30%,#b9b0ff,transparent 35%),linear-gradient(145deg,#6c5ce7,#2e315c 75%);
 box-shadow:0 20px 60px rgba(107,92,231,.35); border:1px solid rgba(255,255,255,.15); }
.eyebrow { display:inline-flex;align-items:center;gap:8px;color:#aaa4ff;font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase; }
.live-dot { width:7px;height:7px;border-radius:50%;background:#55e4b4;box-shadow:0 0 12px #55e4b4;display:inline-block; }
.hero-copy { color:#9199aa;font-size:1.05rem;line-height:1.65;max-width:760px; }
.hero-mini { padding:20px 22px;border-radius:22px;background:linear-gradient(145deg,rgba(139,124,255,.14),rgba(88,224,208,.035));border:1px solid rgba(139,124,255,.20); }
.hero-mini .big { font-family:'Space Grotesk';font-size:38px;font-weight:700;line-height:1; }
.hero-mini .small { color:#7f8798;font-size:11px;letter-spacing:.13em;text-transform:uppercase;margin-bottom:8px; }

/* Uploader. */
div[data-testid="stFileUploaderDropzone"] { min-height:180px!important;display:flex;align-items:center;background:linear-gradient(145deg,rgba(255,255,255,.035),rgba(255,255,255,.012))!important;border:1px dashed rgba(139,124,255,.40)!important;border-radius:20px!important; }
div[data-testid="stFileUploaderDropzone"]:hover { border-color:#8b7cff!important;background:rgba(139,124,255,.055)!important; }

/* Image treatment. */
[data-testid="stImage"] img { border-radius:18px; border:1px solid rgba(255,255,255,.08); }

/* Result typography. */
.result-kicker { color:#8f97a9;font-size:11px;letter-spacing:.17em;text-transform:uppercase;font-weight:700; }
.result-emotion { font-family:'Space Grotesk';font-size:clamp(3rem,5vw,5.2rem);font-weight:700;line-height:.95;letter-spacing:-.055em;margin:8px 0; }
.result-sub { color:#8992a5;font-size:14px; }
.conf-track { height:8px;border-radius:99px;background:#1b202a;overflow:hidden;margin:18px 0 8px; }
.conf-fill { height:100%;border-radius:99px;background:linear-gradient(90deg,#8b7cff,#58e0d0); }

/* Native widgets. */
div[data-testid="stMetric"] { background:rgba(255,255,255,.025);border:1px solid var(--line);border-radius:16px;padding:12px 14px; }
div[data-testid="stProgress"] div[role="progressbar"] { background:#1a1f28;border-radius:99px; }
div[data-testid="stProgress"] div[role="progressbar"] > div { background:linear-gradient(90deg,#8b7cff,#58e0d0)!important;border-radius:99px; }
div[data-testid="stExpander"] { border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.015); }
div[data-testid="stAlert"] { border-radius:15px; }
button[data-baseweb="tab"] { font-weight:700; }

/* Emotion tiles. */
.emotion-tile { padding:17px 15px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.018);min-height:116px; }
.emotion-tile.selected { border-color:rgba(139,124,255,.70);background:linear-gradient(145deg,rgba(139,124,255,.16),rgba(88,224,208,.035));box-shadow:0 0 0 1px rgba(139,124,255,.10),0 18px 40px rgba(50,40,130,.16); }
.emotion-icon { font-size:26px; }
.emotion-name { font-family:'Space Grotesk';font-weight:700;font-size:14px;margin-top:8px; }
.emotion-pct { color:#8e96a8;font-size:12px;margin-top:3px; }

/* Pipeline. */
.pipe-card { position:relative;min-height:154px;padding:20px 16px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.018); }
.pipe-num { color:#6f7788;font-size:11px;letter-spacing:.12em; }
.pipe-title { font-family:'Space Grotesk';font-weight:700;font-size:15px;margin-top:24px; }
.pipe-desc { color:#747d90;font-size:12px;line-height:1.5;margin-top:7px; }
.pipe-line { position:absolute;top:15px;right:-12px;width:24px;height:1px;background:linear-gradient(90deg,#8b7cff,transparent); }

.section-label { color:#70798c;font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;margin-bottom:6px; }
.muted { color:#7d8698; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 Emotion AI")
    st.caption("FACIAL EXPRESSION ANALYSIS")
    st.divider()
    st.markdown("**MODEL**")
    st.write("Deep CNN")
    st.caption("~1.33M parameters · 7 classes")
    st.markdown("**DATASET**")
    st.write("FER2013")
    st.caption("48 × 48 grayscale")
    st.markdown("**RUNTIME**")
    st.write("Apple MPS / CUDA / CPU")
    st.divider()
    auto_detect = st.toggle("Auto-detect face", value=True)
    show_pipeline = st.toggle("Show preprocessing preview", value=False)
    st.divider()
    st.caption("FINAL-YEAR PROJECT")
    st.caption("Deep Learning · Computer Vision")

# ---------------- HERO ----------------
hero_a, hero_b = st.columns([5.2, 1.45], gap="large", vertical_alignment="center")
with hero_a:
    st.markdown('<div class="eyebrow"><span class="live-dot"></span> AI / COMPUTER VISION / FER2013</div>', unsafe_allow_html=True)
    st.markdown("# Human Emotion Detection")
    st.markdown("### Read the expression. Understand the signal.")
    st.markdown('<div class="hero-copy">A deep-learning vision system that transforms a facial image into an interpretable seven-class emotion prediction — from face detection to final confidence.</div>', unsafe_allow_html=True)
with hero_b:
    st.markdown('<div class="hero-orb">🧠</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="hero-mini"><div class="small">MODEL ACCURACY</div><div class="big">65.19%</div></div>', unsafe_allow_html=True)

st.write("")

# ---------------- MODEL LOAD ----------------
with st.spinner("Initializing emotion model…"):
    model, device = load_model()

# ---------------- WORKSPACE ----------------
st.markdown('<div class="section-label">01 / LIVE ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("## Analyze a face")
st.caption("Upload a clear front-facing image or use your camera. The trained Deep CNN performs the actual classification.")

with st.container(border=True):
    upload_tab, camera_tab = st.tabs(["📁  UPLOAD IMAGE", "📷  CAPTURE PHOTO"])
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

# ---------------- RESULT ----------------
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
        accent = EMOTION_ACCENT[emotion]

        st.write("")
        st.markdown('<div class="section-label">02 / INFERENCE RESULT</div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 1.65], gap="large", vertical_alignment="center")
        with left:
            with st.container(border=True):
                st.markdown('<div class="result-kicker">SOURCE FRAME</div>', unsafe_allow_html=True)
                st.image(active_img, use_container_width=True)
                if auto_detect:
                    st.success("Face detected and isolated")
                else:
                    st.info("Auto-detection disabled")
        with right:
            with st.container(border=True):
                st.markdown('<div class="result-kicker">DETECTED EMOTION</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-emotion" style="color:{accent}">{EMOTION_EMOJI[emotion]} {emotion}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-sub">The model assigns the highest probability to <b>{emotion}</b>.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="conf-track"><div class="conf-fill" style="width:{conf*100:.1f}%"></div></div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("CONFIDENCE", f"{conf*100:.1f}%")
                with c2:
                    st.metric("DEVICE", str(device).upper())
                if conf < 0.45:
                    st.warning("Low confidence — a clearer, front-facing image may improve the result.")

        st.write("")
        st.markdown('<div class="section-label">03 / PROBABILITY MAP</div>', unsafe_allow_html=True)
        top_col, grid_col = st.columns([1.15, 2.25], gap="large")
        with top_col:
            with st.container(border=True):
                st.markdown("### Top predictions")
                st.caption("Ranked output from the seven-class classifier")
                for rank, idx in enumerate(top3, 1):
                    name = CLASS_NAMES[idx]
                    p = float(probs[idx])
                    st.write(f"**{rank:02d} · {EMOTION_EMOJI[name]} {name}**")
                    st.progress(p)
                    st.caption(f"{p*100:.1f}%")
        with grid_col:
            st.markdown("### Emotion landscape")
            cols = st.columns(4)
            for i, name in enumerate(CLASS_NAMES):
                p = float(probs[i])
                selected = i == pred_idx
                card_class = "emotion-tile selected" if selected else "emotion-tile"
                with cols[i % 4]:
                    st.markdown(
                        f'<div class="{card_class}"><div class="emotion-icon">{EMOTION_EMOJI[name]}</div>'
                        f'<div class="emotion-name">{name}</div><div class="emotion-pct">{p*100:.1f}%</div></div>',
                        unsafe_allow_html=True,
                    )

        if show_pipeline:
            st.write("")
            st.markdown('<div class="section-label">04 / PREPROCESSING PREVIEW</div>', unsafe_allow_html=True)
            preview = pipeline_preview_images(face_img)
            pcols = st.columns(3)
            for col, (caption, im) in zip(pcols, preview):
                with col:
                    with st.container(border=True):
                        st.image(im, use_container_width=True)
                        st.caption(caption)

# ---------------- PIPELINE ----------------
st.write("")
st.markdown('<div class="section-label">05 / PIPELINE</div>', unsafe_allow_html=True)
st.markdown("## From pixels to prediction")
st.caption("The same trained inference pipeline is used for every analysis.")
pipe_cols = st.columns(len(PIPELINE_STEPS))
for i, (col, (num, title, desc)) in enumerate(zip(pipe_cols, PIPELINE_STEPS)):
    with col:
        arrow = '<div class="pipe-line"></div>' if i < len(PIPELINE_STEPS)-1 else ''
        st.markdown(f'<div class="pipe-card"><div class="pipe-num">{num}</div><div class="pipe-title">{title}</div><div class="pipe-desc">{desc}</div>{arrow}</div>', unsafe_allow_html=True)

# ---------------- MODEL / DATASET ----------------
st.write("")
st.markdown('<div class="section-label">06 / SYSTEM PROFILE</div>', unsafe_allow_html=True)
info_a, info_b, info_c = st.columns([1.25, 1.25, 1], gap="large")
with info_a:
    with st.container(border=True):
        st.caption("ARCHITECTURE")
        st.markdown("## Deep CNN")
        st.write("Custom convolutional architecture with double-convolution blocks, SE attention, global average pooling and a fully connected classifier.")
        st.metric("Parameters", "~1.33M")
with info_b:
    with st.container(border=True):
        st.caption("DATASET")
        st.markdown("## FER2013")
        st.write("Facial expression dataset used to train and evaluate the seven-class emotion recognition system.")
        st.metric("Input", "48 × 48 grayscale")
with info_c:
    with st.container(border=True):
        st.caption("PERFORMANCE")
        st.markdown("## 65.19%")
        st.write("Test accuracy of the selected Deep CNN model.")
        st.metric("Classes", "7 emotions")

with st.expander("📘  About this project"):
    st.write("Human facial expressions provide visual cues about emotional states. This project combines OpenCV face detection, image enhancement and a custom Deep CNN to classify facial expressions into Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.")

st.divider()
fa, fb = st.columns([3, 1])
with fa:
    st.caption("HUMAN EMOTION DETECTION  ·  DEEP LEARNING  ·  FER2013")
with fb:
    st.caption(f"RUNTIME  {str(device).upper()}")
