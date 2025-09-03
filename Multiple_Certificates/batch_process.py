import os
import asyncio
import concurrent.futures
import pdf2image
import json
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from data_extractor import DataExtractor
from Bbox_OCR import BoundingBoxOCR
from ocr_final import OCR_JSON
import argparse
import tempfile

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
poppler_path = os.path.abspath(os.path.join(base_dir, "..", "Streamlit", "assets", "poppler_bin"))
output_excel_path = "batch_processed_certificates.xlsx"
annotated_output_dir = "annotated_output"

# --- Fields for final Excel ---
REQUIRED_FIELDS = {
    "country_name": "Country",
    "validity_in_months": "Validity",
    "Model_number": "Model code",
    "certificate_issue_date": "Last approval date",
    "certificate_expiry_date": "Certification expiry date"
}

def find_all_pdfs(folder):
    pdf_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths

def enhance_images(images, zoom=4.5):
    enhanced = []
    for img in images:
        w, h = img.size
        enhanced_img = img.resize((int(w * zoom), int(h * zoom)), Image.LANCZOS)
        enhanced.append(enhanced_img)
    return enhanced

def annotate_images(image_paths, polygons):
    annotated_imgs = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        overlay = img.copy()
        text_positions = []
        page_polygons = [p for p in polygons if p["page"] == idx + 1]
        for p in page_polygons:
            pts = np.array([[pt['x'], pt['y']] for pt in p['polygon']], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (255, 120, 0))
            text_positions.append((tuple(pts[0][0]), p['label']))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        for pos, label in text_positions:
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)

        annotated_imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return annotated_imgs

async def process_pdf(pdf_path, input_root, enhance_images_flag=True):
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        images = pdf2image.convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
        if enhance_images_flag:
            images = enhance_images(images)

        # Save temporary images
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for i, img in enumerate(images):
                path = os.path.join(tmpdir, f"page_{i+1}.jpg")
                img.save(path)
                image_paths.append(path)

            # Run both tasks in parallel
            loop = asyncio.get_event_loop()
            executor = concurrent.futures.ThreadPoolExecutor()

            async def run_data_extraction():
                return await loop.run_in_executor(executor, lambda: DataExtractor().extract_data(images))

            async def run_ocr_processing():
                return await loop.run_in_executor(executor, lambda: OCR_JSON().process_images(image_paths))

            data_result, ocr_result = await asyncio.gather(
                run_data_extraction(),
                run_ocr_processing()
            )

            result_dict = json.loads(data_result)
            required_data = result_dict.get("required", {})
            extracted_data = result_dict.get("extracted", {})
            processed_df = pd.DataFrame([extracted_data])
            country_value = required_data.get("country_name", "")

            polygons = BoundingBoxOCR().extract_box(ocr_result, processed_df, country_value)
            annotated_imgs = annotate_images(image_paths, polygons)

            # Save annotated PDF in mirrored structure
            relative_path = os.path.relpath(pdf_path, input_root)
            original_stem = Path(relative_path).stem
            parent_dir = Path(relative_path).parent
            annotated_filename = f"{original_stem}_annotated.pdf"
            annotated_pdf_path = os.path.join(annotated_output_dir, parent_dir, annotated_filename)

            os.makedirs(os.path.dirname(annotated_pdf_path), exist_ok=True)
            annotated_imgs[0].save(annotated_pdf_path, save_all=True, append_images=annotated_imgs[1:])

            # Prepare Excel row
            formatted_data = {v: required_data.get(k, "") for k, v in REQUIRED_FIELDS.items()}
            formatted_data["source_file"] = os.path.basename(pdf_path)
            rel_path = os.path.relpath(annotated_pdf_path, os.path.dirname(output_excel_path))
            display_text = os.path.basename(annotated_pdf_path)  

            formatted_data["annotated_pdf_path"] = f'=HYPERLINK("{rel_path}", "{annotated_pdf_path}")'

            return formatted_data

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return {
            "source_file": os.path.basename(pdf_path),
            "annotated_pdf_path": "ERROR",
            **{v: "ERROR" for v in REQUIRED_FIELDS.values()}
        }

async def batch_process(folder_path, enhance=True):
    pdf_files = find_all_pdfs(folder_path)
    print(f"Found {len(pdf_files)} PDF files to process.")

    tasks = [process_pdf(pdf_path, input_root=folder_path, enhance_images_flag=enhance) for pdf_path in pdf_files]
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"\n✅ Batch processing complete. Output saved to: {output_excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process certificates and export annotated PDFs.")
    parser.add_argument("input_folder", type=str, help="Path to folder containing PDF files or folders")
    parser.add_argument("--no-enhance", action="store_true", help="Disable image enhancement")
    args = parser.parse_args()

    asyncio.run(batch_process(args.input_folder, enhance=not args.no_enhance))
