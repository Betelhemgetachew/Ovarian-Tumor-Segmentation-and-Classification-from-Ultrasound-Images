import base64
import io
import os

import albumentations as A
import cv2
import gdown
import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from fpdf import FPDF
from PIL import Image

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Ovarian Ultrasound Analysis",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="auto",
)

# =========================
# Styling (colours also set in .streamlit/config.toml)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }

.app-head h1 { font-size: 1.7rem; font-weight: 700; margin: 0; color: #101828; }
.app-head p { color: #667085; margin: 0.25rem 0 0; }
.notice { background: #fdf3e1; border: 1px solid #f1d9a7; border-left: 3px solid #b7791f; color: #7a5214;
          border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 0.86rem; margin: 1rem 0 1.5rem; }

.card { background: #fff; border: 1px solid #e4e7ec; border-radius: 10px; padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(16,24,40,0.05); height: 100%; }
.card h4 { margin: 0 0 0.6rem; font-size: 0.95rem; font-weight: 600; color: #101828; }
.label { font-size: 0.8rem; font-weight: 500; color: #475467; }
.class-name { font-size: 1.6rem; font-weight: 700; line-height: 1.25; margin: 0.2rem 0 0.4rem; color: #101828; }
.tag { display: inline-block; border-radius: 999px; padding: 0.15rem 0.65rem; font-size: 0.78rem; font-weight: 600; }
.tag-benign { background: #e7f4ec; color: #2f855a; }
.tag-malignant { background: #fdeceb; color: #c0392b; }
.tag-normal { background: #eaf1fe; color: #2563eb; }
.tag-low { background: #fdf3e1; color: #b7791f; }
.big { font-size: 2rem; font-weight: 700; color: #101828; font-variant-numeric: tabular-nums; }
.desc { color: #475467; font-size: 0.92rem; margin-top: 0.75rem; line-height: 1.55; }

.prob { display: grid; grid-template-columns: 12.5rem 1fr 3.5rem; align-items: center; gap: 0.75rem; margin: 0.45rem 0; font-size: 0.88rem; }
.prob .name { color: #101828; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prob .track { background: #f2f4f7; border-radius: 99px; height: 8px; overflow: hidden; }
.prob .fill { height: 8px; border-radius: 99px; background: #0e6e6a; }
.prob.top .name { font-weight: 600; }
.prob .val { text-align: right; color: #475467; font-variant-numeric: tabular-nums; }

.figure img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; border-radius: 8px; border: 1px solid #e4e7ec; background: #000; }
.figure .cap { font-size: 0.82rem; color: #667085; margin-top: 0.4rem; }
.empty { border: 1px dashed #d0d5dd; border-radius: 10px; padding: 2.5rem 1.5rem; text-align: center; color: #475467; background: #fff; }
.empty b { color: #101828; }
.foot { color: #98a2b3; font-size: 0.8rem; margin-top: 3rem; border-top: 1px solid #e4e7ec; padding-top: 1rem; }
.stDownloadButton button, .stButton button { border-radius: 8px; font-weight: 600; }
@media (max-width: 640px) { .prob { grid-template-columns: 9rem 1fr 3rem; font-size: 0.82rem; } .class-name { font-size: 1.35rem; } .big { font-size: 1.6rem; } }
</style>
""", unsafe_allow_html=True)

# -------- Device --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Configuration & Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")  # weights are in the repo; downloaded from Drive only if missing

# Ensure models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

SEG_MODEL_PATH = os.path.join(MODEL_DIR, "mmotu_segmentation_best_model.pth")
CLS_MODEL_PATH = os.path.join(MODEL_DIR, "mmotu_classifier_fullimage_best.pth")

# --- GOOGLE DRIVE FILE IDs ---
SEG_FILE_ID = '1JL8mDDxJ81dhF9s3sSV2JOxlZglsSCaF' 
CLS_FILE_ID = '1kLNPFI1nKd3cN-P57HX7tgOv3kZnb7Sy'
# -----------------------------

CLASS_NAMES = [  # order must match the MMOTU label IDs 0-7 in train_cls.txt / val_cls.txt
    "Chocolate Cyst", "Serous Cystadenoma", "Teratoma", "Theca Cell Tumor",
    "Simple Cyst", "Normal Ovary", "Mucinous Cystadenoma", "High-Grade Serous Carcinoma"
]
NUM_CLASSES = len(CLASS_NAMES)

# --- Medical Dictionary ---
MEDICAL_INFO = {
    "Normal Ovary": "Healthy ovarian tissue with no visible abnormalities.",
    "Simple Cyst": "A fluid-filled sac. Most are benign and often resolve on their own.",
    "Chocolate Cyst": "Endometrioma. A cyst filled with old blood, often linked to endometriosis.",
    "Serous Cystadenoma": "A benign tumor filled with thin, watery fluid.",
    "Mucinous Cystadenoma": "A benign tumor filled with a gelatinous, mucous-type fluid.",
    "Teratoma": "A germ cell tumor that may contain hair, teeth, or bone tissue. Mature cystic teratomas are usually benign.",
    "Theca Cell Tumor": "A rare, usually benign solid tumor of the ovary.",
    "High-Grade Serous Carcinoma": "A type of ovarian cancer that requires immediate clinical attention."
}

# =========================
# 2. PDF Generation Function
# =========================
def create_pdf(diagnosis, confidence, image_path, notes=""):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(190, 15, txt="MMOTU Research Demo: Model Output", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(190, 10, txt="Automated Ultrasound Analysis", ln=1, align='C')
    pdf.line(10, 35, 200, 35)
    
    # Image
    pdf.ln(10)
    pdf.image(image_path, x=55, y=45, w=100)
    
    # Results
    pdf.ln(110)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, txt="Results", ln=1, align='L')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(50, 10, txt="Predicted class:", align='L')
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(140, 10, txt=f"{diagnosis}", ln=1, align='L')
    
    # Add Definition
    definition = MEDICAL_INFO.get(diagnosis, "")
    if definition:
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(190, 6, txt=f"({definition})")
        pdf.ln(2)

    pdf.set_font("Arial", size=12)
    pdf.cell(50, 10, txt="Confidence:", align='L')
    pdf.cell(140, 10, txt=f"{confidence:.2f}%", ln=1, align='L')
    
    if notes:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(190, 10, txt="Notes:", ln=1, align='L')
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(190, 10, txt=notes)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(190, 5, txt="DISCLAIMER: This report is generated by an AI system (MMOTU Project). It is intended for research and educational purposes only.")
    
    return pdf.output(dest="S").encode("latin-1")

# =========================
# 3. Load Models (Spinner / Invisible Download)
# =========================
@st.cache_resource
def load_models():
    # Only show spinner if models are missing
    if not os.path.exists(SEG_MODEL_PATH) or not os.path.exists(CLS_MODEL_PATH):
        with st.spinner("Setting up AI models... (This happens once)"):
            if not os.path.exists(SEG_MODEL_PATH):
                url = f'https://drive.google.com/uc?id={SEG_FILE_ID}'
                gdown.download(url, SEG_MODEL_PATH, quiet=False)

            if not os.path.exists(CLS_MODEL_PATH):
                url = f'https://drive.google.com/uc?id={CLS_FILE_ID}'
                gdown.download(url, CLS_MODEL_PATH, quiet=False)

    seg_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid")
    cls_model = models.resnet18(weights=None)
    cls_model.fc = nn.Linear(cls_model.fc.in_features, NUM_CLASSES)

    try:
        if os.path.exists(SEG_MODEL_PATH):
            seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
        else:
            st.error(f"❌ Segmentation model failed to download.")
            return None, None

        if os.path.exists(CLS_MODEL_PATH):
            cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
        else:
            st.error(f"❌ Classifier model failed to download.")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

    seg_model.to(DEVICE).eval()
    cls_model.to(DEVICE).eval()
    return seg_model, cls_model

seg_model, cls_model = load_models()

transform_base = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# =========================
# 4. Processing Functions
# =========================
def process_segmentation(pil_image, threshold=0.5):
    img = np.array(pil_image.convert("RGB"))
    original_size = img.shape[:2]

    augmented = transform_base(image=img)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask = seg_model(input_tensor)[0,0].cpu().numpy()

    mask_bin = (pred_mask > threshold).astype(np.uint8)
    mask_resized = cv2.resize(mask_bin, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(img)
    color_mask[:, :, 0] = 255 
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask_resized)
    overlay = cv2.addWeighted(img, 1.0, color_mask, 0.4, 0)

    return mask_resized, overlay

def crop_tumor_region(pil_image, mask_bin):
    img_arr = np.array(pil_image.convert("RGB"))
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None 

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = 20
    h_img, w_img = img_arr.shape[:2]
    y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
    x1, x2 = max(0, x - pad), min(w_img, x + w + pad)

    cropped_img = img_arr[y1:y2, x1:x2]
    return Image.fromarray(cropped_img)

def predict_class(pil_image):
    img = np.array(pil_image.convert("RGB"))
    augmented = transform_base(image=img)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = cls_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
    return probs

# =========================
# 5. Helpers for the interface
# =========================
MALIGNANT = {"High-Grade Serous Carcinoma"}
LOW_CONFIDENCE = 0.60


def to_data_uri(img):
    """PIL image or RGB array -> base64 PNG data URI for inline HTML."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(img, caption):
    st.markdown(f'<div class="figure"><img src="{to_data_uri(img)}" alt="{caption}"><div class="cap">{caption}</div></div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def analyse(image_bytes, threshold):
    """Run segmentation and classification once per image and threshold."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask, overlay = process_segmentation(image, threshold=threshold)
    crop = crop_tumor_region(image, mask) if mask.sum() > 0 else None
    probs = predict_class(image)
    return mask, overlay, crop, probs


# =========================
# 6. Layout
# =========================
st.markdown("""
<div class="app-head">
  <h1>Ovarian ultrasound analysis</h1>
  <p>Lesion segmentation (U-Net) and tumour classification (ResNet18) on 2D ultrasound images.</p>
</div>
<div class="notice">Research demo trained on the MMOTU dataset. It is not a diagnostic tool and must not be used for patient care.</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    seg_threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Lower values outline more of the image; higher values keep only confident lesion pixels.")
    st.caption("Lower it if the outline misses part of the lesion; raise it if the outline spills into background.")
    st.markdown("---")
    st.markdown("### About the models")
    st.markdown(
        "- **Segmentation:** U-Net with a ResNet34 encoder. Validation Dice 0.85, IoU 0.77.\n"
        "- **Classification:** ResNet18, 8 categories. Validation accuracy 71.9%.\n"
        "- **Data:** MMOTU, 1,469 ultrasound images (Zhao et al., 2022)."
    )
    st.markdown("---")
    st.caption("Built by Betelhem Getachew Kebede")

uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"],
                                 help="A 2D ovarian ultrasound image, JPG or PNG.")

if not (seg_model and cls_model):
    st.error("The models could not be loaded. Check that the weights are in the Models folder.")
elif uploaded_file is None:
    st.markdown("""
    <div class="empty">
      <b>Upload a 2D ovarian ultrasound image to start.</b><br>
      The app outlines the lesion, shows a close-up, and predicts one of 8 categories:
      chocolate cyst, serous cystadenoma, teratoma, theca cell tumour, simple cyst,
      normal ovary, mucinous cystadenoma or high-grade serous carcinoma.
    </div>
    """, unsafe_allow_html=True)
else:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with st.spinner("Analysing the image ..."):
        mask, overlay, crop, probs = analyse(image_bytes, seg_threshold)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    diagnosis = CLASS_NAMES[top_idx]
    uncertain = confidence < LOW_CONFIDENCE
    lesion_share = float(mask.mean()) * 100

    # --- Images ---
    st.markdown("#### Images")
    c1, c2, c3 = st.columns(3)
    with c1:
        figure(image, "Uploaded image")
    with c2:
        figure(overlay, f"Predicted lesion outline (threshold {seg_threshold:.2f})")
    with c3:
        if crop is not None:
            figure(crop, "Close-up of the largest lesion region")
        else:
            st.markdown('<div class="empty">No lesion region found at this threshold. Try lowering it in the sidebar.</div>',
                        unsafe_allow_html=True)

    # --- Results ---
    st.markdown("#### Results")
    r1, r2 = st.columns([1, 1.35], gap="medium")
    with r1:
        if uncertain:
            tag = '<span class="tag tag-low">Low confidence</span>'
        elif diagnosis in MALIGNANT:
            tag = '<span class="tag tag-malignant">Malignant category</span>'
        elif diagnosis == "Normal Ovary":
            tag = '<span class="tag tag-normal">No tumour</span>'
        else:
            tag = '<span class="tag tag-benign">Benign category</span>'
        note = ("The top prediction is below 60% confidence, so treat it as uncertain."
                if uncertain else MEDICAL_INFO.get(diagnosis, ""))
        st.markdown(f"""
        <div class="card">
          <div class="label">Predicted class</div>
          <div class="class-name">{diagnosis}</div>
          {tag}
          <div style="display:flex;gap:2.5rem;margin-top:1rem">
            <div><div class="label">Confidence</div><div class="big">{confidence*100:.1f}%</div></div>
            <div><div class="label">Lesion area</div><div class="big">{lesion_share:.1f}%</div></div>
          </div>
          <div class="desc">{note}</div>
          <div class="desc" style="font-size:0.8rem;color:#98a2b3">Lesion area is the share of the image inside the predicted outline. The classifier uses the whole image.</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        order = np.argsort(probs)[::-1]
        rows = "".join(
            f'<div class="prob{" top" if i == top_idx else ""}"><span class="name">{CLASS_NAMES[i]}</span>'
            f'<span class="track"><span class="fill" style="width:{probs[i]*100:.1f}%;display:block"></span></span>'
            f'<span class="val">{probs[i]*100:.1f}%</span></div>'
            for i in order
        )
        st.markdown(f'<div class="card"><h4>Probability for each category</h4>{rows}</div>', unsafe_allow_html=True)

    # --- Export ---
    st.markdown("#### Export")
    e1, e2 = st.columns([2, 1])
    with e1:
        notes = st.text_area("Notes to include in the PDF (optional)", height=90)
    with e2:
        final_status = f"Uncertain ({diagnosis}?)" if uncertain else diagnosis
        temp_img_path = "temp_scan_report.png"
        image.save(temp_img_path)
        pdf_bytes = create_pdf(final_status, confidence * 100, temp_img_path, notes)
        os.remove(temp_img_path)
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        st.download_button("Download PDF summary", data=pdf_bytes, file_name="ultrasound_model_output.pdf",
                           mime="application/pdf", type="primary", use_container_width=True)

st.markdown('<div class="foot">Built by Betelhem Getachew Kebede. '
            'Data: MMOTU dataset (Zhao et al., 2022).</div>', unsafe_allow_html=True)
