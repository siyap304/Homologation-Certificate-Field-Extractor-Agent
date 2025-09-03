# --- Imports ---
import os
import asyncio
import concurrent.futures
import streamlit as st
import pdf2image
import io
import json
import pandas as pd
from data_extractor import DataExtractor
from Bbox_OCR import BoundingBoxOCR
import cv2
import tempfile
import numpy as np
from ocr_final import OCR_JSON
from PIL import Image
import base64
import webbrowser
from pathlib import Path

# --- Configuration ---
poppler_path = os.environ.get("POPLER_BIN_PATH", None)
manual_path = os.environ.get("USER_MANUAL_PATH", None)
logo_path = os.environ.get("LOGO_PATH", None)
if not poppler_path:
    raise RuntimeError("Poppler binary path not set")
if not manual_path:
    raise RuntimeError("User Manual path not set")

# Page configuration
st.set_page_config(layout="wide", page_title="Certificate Processor")

st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- CSS Styling ---
st.markdown("""
    <style>
    div.stForm button {
        height:25px !important;
        width: 100% !important;
        font-size: 30px !important;
        font-weight: bold !important; 
        background-color: #0096FF !important;
        color: white !important;
        transition: background-color 0.3s ease;
    }
    div.stForm button:hover {
        background-color: #0072cc !important; 
        cursor: pointer;
    }

    .custom-title {
        font-size: 27px !important;
        font-weight: 700 !important;
        color: #E0E0E0 !important;
        padding-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image(logo_path)

st.markdown(f"""
    <div style="
        position: fixed;
        bottom: 20px;
        right: 20px;
    ">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 100px; height: auto;">
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style="
        margin-top: 0px;
        text-align: right;
        font-size: 16px;
        font-style: italic;
        font-weight: 600;
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    ">
        Developed By AI Solutions Team, HIA-Pune
    </div>
""", unsafe_allow_html=True)

# --- Title and Button Layout ---
col1, col2 = st.columns([5, 0.7])

with col1:
    st.markdown('<div class="custom-title">HOMOLOGATION CERTIFICATE PROCESSING AND DATA EXPORT</div>', unsafe_allow_html=True)
with col2:
    if st.button("📘 User Manual"):
        webbrowser.open(f"file://{manual_path}")

# --- Session State ---
for key in [
    "pdf_images", "enhanced_images", "annotated_images",
    "processed_df", "required_data", "preview_mode", "extracted_data", "zoom_level"
]:
    if key not in st.session_state:
        st.session_state[key] = None
query_params = st.query_params  

# --- File Upload ---
with st.container():
    cols = st.columns([4.5, 4, 4])

    uploaded_file = cols[0].file_uploader("Upload Certificate", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        if "last_uploaded_filename" not in st.session_state or uploaded_file.name != st.session_state["last_uploaded_filename"]:
            pdf_bytes = uploaded_file.read()
            st.session_state["pdf_images"] = pdf2image.convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
            st.session_state["enhanced_images"] = None
            st.session_state["annotated_images"] = None
            st.session_state["preview_mode"] = "original"
            st.session_state["extracted_data"] = None
            st.session_state["required_data"] = None
            st.session_state["last_uploaded_filename"] = uploaded_file.name

    # --- Process Certificate ---
    with cols[1].form(key="process_form"):
        process_button = st.form_submit_button("⚙️ Process Certificate")

    if process_button and (st.session_state.get("pdf_images") or st.session_state.get("enhanced_images")):
        async def process_certificate():
            preview_mode = st.session_state["preview_mode"]
            if preview_mode == "enhanced":
                images = st.session_state["enhanced_images"]
                st.write("Using ENHANCED certificate for processing.")
            else:
                images = st.session_state["pdf_images"]
                st.write("Using ORIGINAL certificate for processing.")

            # Save images to temp files
            with tempfile.TemporaryDirectory() as tmpdir:
                image_paths = []
                for i, img in enumerate(images):
                    path = os.path.join(tmpdir, f"page_{i+1}.jpg")
                    img.save(path)
                    image_paths.append(path)

                loop = asyncio.get_event_loop()
                executor = concurrent.futures.ThreadPoolExecutor()

                async def run_data_extraction():
                    return await loop.run_in_executor(executor, lambda: DataExtractor().extract_data(images))

                async def run_ocr_json():
                    return await loop.run_in_executor(executor, lambda: OCR_JSON().process_images(image_paths))

                data_result, ocr_result = await asyncio.gather(run_data_extraction(), run_ocr_json())
                result_dict = json.loads(data_result)
                st.session_state["required_data"] = result_dict.get("required", {})
                country_value = st.session_state["required_data"].get("country_name", None)
                st.session_state["processed_df"] = pd.DataFrame([result_dict.get("extracted", {})])
                st.session_state["extracted_data"] = result_dict.get("extracted", {})

                extractor = BoundingBoxOCR()
                polygons = extractor.extract_box(ocr_result, st.session_state["processed_df"], country_value)

                annotated = {}
                for idx, img_path in enumerate(image_paths):
                    img = cv2.imread(img_path)
                    overlay = img.copy()
                    text_positions = []
                    page_polygons = [p for p in polygons if p["page"] == idx + 1]
                    for p in page_polygons:
                        pts = np.array([[pt['x'], pt['y']] for pt in p['polygon']], dtype=np.int32).reshape((-1, 1, 2))
                        color = (0, 120, 255)
                        cv2.fillPoly(overlay, [pts], color)
                        text_positions.append((tuple(pts[0][0]), p['label']))
                    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                    font_scale = 0.7 if preview_mode != "enhanced" else 0.7 * 4.5
                    font_thickness = 2 if preview_mode != "enhanced" else int(2 * 4.5)
                    for pos, label in text_positions:
                        cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 139), font_thickness)

                    annotated[idx + 1] = img

                st.session_state["annotated_images"] = annotated
                st.session_state["preview_mode"] = "annotated"

        asyncio.run(process_certificate())

    # --- Enhance Certificate ---
    with cols[2].form(key="enhance_form"):
        enhance_button = st.form_submit_button("🔧 Enhance Certificate")

    if enhance_button and st.session_state.get("pdf_images"):
        zoom = 4.5
        enhanced_images = []
        for img in st.session_state["pdf_images"]:
            width, height = img.size
            enhanced_img = img.resize((int(width * zoom), int(height * zoom)), Image.LANCZOS)
            enhanced_images.append(enhanced_img)
        st.session_state["enhanced_images"] = enhanced_images
        st.session_state["preview_mode"] = "enhanced"
        st.session_state["extracted_data"] = None
        st.session_state["required_data"] = None

# --- Display Left: Processed Data ---
left_col, right_col = st.columns([1, 2])
with left_col:
    st.subheader("📋 Processing Output")

    if st.session_state["required_data"]:
        st.markdown("### Required Format")
        fields = {
            "country_name": "Country",
            "validity_in_months": "Validity",
            "Model_number": "Model code",
            "certificate_issue_date": "Last approval date",
            "certificate_expiry_date": "Certification expiry date"
        }
        filtered = {v: st.session_state["required_data"].get(k, "") for k, v in fields.items()}
        df_csv = pd.DataFrame([filtered])
        df_transposed = df_csv.T
        df_transposed.columns = ["Value(Edit if required)"]
        df_transposed.index.name = "Field"
        edited_df = st.data_editor(df_transposed, num_rows="fixed", use_container_width=True)
        df_for_download = edited_df.T

        towrite = io.BytesIO()
        df_for_download.to_excel(towrite, index=False, engine="openpyxl")
        towrite.seek(0)

        st.markdown("""
        <style>
        div.stDownloadButton > button {
            background-color: #0096FF;
            color: white;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 16px;
        }
        div.stDownloadButton > button:hover {
            background-color: #0072cc;
        }
        </style>
        """, unsafe_allow_html=True)

        st.download_button("📥 Download Excel", data=towrite, file_name="processed_certificate.xlsx")

    if st.session_state["extracted_data"]:
        st.markdown("### Actual Data")
        data_dict = st.session_state["extracted_data"]
        df = pd.DataFrame(list(data_dict.items()), columns=["Field", "Value"])
        st.table(df.set_index("Field"))

# --- Display Right: Certificate Preview ---
with right_col:
    cols = st.columns([4, 2, 2])
    cols[0].subheader("📄 Certificate Preview")

    zoom_level = cols[0].slider(
        "Zoom Level (%)", min_value=100, max_value=200, step=10,
        value=st.session_state.get("zoom_level", 100)
    )
    st.session_state["zoom_level"] = zoom_level

    # --- Rotate Helpers ---
    def rotate_images(images, degrees):
        return [img.rotate(degrees, expand=True) for img in images]

    def rotate_annotated_images(images_dict, degrees):
        rotated_dict = {}
        for page_num, img in images_dict.items():
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            rotated_pil = pil_img.rotate(degrees, expand=True)
            rotated_cv = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)
            rotated_dict[page_num] = rotated_cv
        return rotated_dict

    with cols[1].form("rotate_clock"):
        rotate_clock = st.form_submit_button("Rotate ↻")
    if rotate_clock:
        mode = st.session_state.get("preview_mode")
        if mode == "original":
            st.session_state["pdf_images"] = rotate_images(st.session_state["pdf_images"], -90)
        elif mode == "enhanced":
            st.session_state["enhanced_images"] = rotate_images(st.session_state["enhanced_images"], -90)
        elif mode == "annotated":
            st.session_state["annotated_images"] = rotate_annotated_images(st.session_state["annotated_images"], -90)
            st.session_state["pdf_images"] = rotate_images(st.session_state["pdf_images"], -90)

    with cols[2].form("rotate_anti"):
        rotate_anti = st.form_submit_button("Rotate ↺")
    if rotate_anti:
        mode = st.session_state.get("preview_mode")
        if mode == "original":
            st.session_state["pdf_images"] = rotate_images(st.session_state["pdf_images"], 90)
        elif mode == "enhanced":
            st.session_state["enhanced_images"] = rotate_images(st.session_state["enhanced_images"], 90)
        elif mode == "annotated":
            st.session_state["annotated_images"] = rotate_annotated_images(st.session_state["annotated_images"], 90)
            st.session_state["pdf_images"] = rotate_images(st.session_state["pdf_images"], 90)

    # --- Image Viewer ---
    def render_images(images, preview_mode, zoom_level):
        st.write(f"Using {preview_mode} certificate — Zoom: {zoom_level}%")
        scale = zoom_level / 100.0
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            st.markdown(f"""
                <div style="overflow: auto; border: 1px solid #ccc; margin-bottom: 16px; max-height: 100vh;">
                    <div style="transform: scale({scale}); transform-origin: top left; width: fit-content;">
                        <img src="data:image/png;base64,{img_b64}" style="display: block;">
                    </div>
                </div>
            """, unsafe_allow_html=True)

    preview_mode = st.session_state["preview_mode"]
    if preview_mode == "annotated" and st.session_state["annotated_images"]:
        imgs = [st.session_state["annotated_images"][i] for i in sorted(st.session_state["annotated_images"])]
        render_images(imgs, preview_mode, zoom_level)
    elif preview_mode == "enhanced" and st.session_state["enhanced_images"]:
        render_images(st.session_state["enhanced_images"], preview_mode, zoom_level)
    elif preview_mode == "original" and st.session_state["pdf_images"]:
        render_images(st.session_state["pdf_images"], preview_mode, zoom_level)
