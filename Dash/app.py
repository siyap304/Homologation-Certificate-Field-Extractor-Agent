import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import base64
import pdf2image
import io
import json
import pandas as pd
import fitz
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
import asyncio
import concurrent.futures
from collections import defaultdict
from io import BytesIO
from utils.data_extractor import DataExtractor
from utils.Bbox_OCR import BoundingBoxOCR
from utils.ocr_final import OCR_JSON

base_dir = os.path.dirname(os.path.abspath(__file__))
poppler_path = os.path.abspath(os.path.join(base_dir, "..", "Streamlit", "assets", "poppler_bin"))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app._callback_list = []

app.layout = dbc.Container([
    html.Div([
        html.Img(
            src="/assets/AIS_Logo.png",
            style={
                "height": "40px",
                "position": "absolute",
                "left": "20px",
                "top": "50%",
                "transform": "translateY(-50%)"
            }
        ),
        html.H3(
            "Homologation Certificate Processing and Data Export",
            style={
            "textAlign": "center",
            "margin": 0,
            "width": "100%"
            }
        ),
        html.Div(
            "Developed By AI Solutions Team, HIA-Pune",
            style={
                "fontSize": "0.95rem",
                "font-style": "italic",
                "color": "white",
                "position": "absolute",
                "bottom": "5px",
                "right": "10px"
            }
        )
    ],
    style={
        "backgroundColor": "#0B1F3A",
        "borderRadius": "5px",
        "color": "white",
        "padding": "20px",
        "position": "relative"
    }),

    dbc.Row([
        dbc.Col(dcc.Upload(
            id='upload-certificate',
            children=dbc.Button("Upload Certificate", style={"backgroundColor": "#3A5D85", "color": "white", "border": "none"}, className="mt-2"),
            multiple=False,
            style={"textAlign": "center"}
        ), width="auto"),
        dbc.Col(dbc.Button("Process Certificate", id="process-certificate", style={"backgroundColor": "#3A5D85", "color": "white", "border": "none"}, className="mt-2"), width="auto"),
        dbc.Col(dbc.Button("Enhance Certificate", id="enhance-certificate", style={"backgroundColor": "#3A5D85", "color": "white", "border": "none"}, className="mt-2"), width="auto"),
        dbc.Col(html.A(
        dbc.Button("User Manual", style={"backgroundColor": "#3A5D85", "color": "white", "border": "none"}, className="mt-2"),
        href="/assets/UserManual.pdf",
        target="_blank"
    ), width="auto"),
    ], className="mt-3 justify-content-center g-2"),

    dbc.Row([
        dbc.Col([
            html.H4("Processed Data", className="text-center text-primary mt-3"),
            dcc.Loading(
                id="loading-processing",
                type="default",
                children=[
                    html.Div(id="processing-status", className="text-center text-warning mt-2"),
                    html.Div(id="processed-data-container", className="border p-3", style={"backgroundColor": "#f8f9fa", "borderRadius": "5px"})
                ]
            )
        ], width=4),
        
        dbc.Col([
            html.H4("Certificate Preview", className="text-center text-primary mt-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Slider(
                        id="zoom-slider",
                        min=100,
                        max=200,
                        step=10,
                        value=100,
                        marks={i: f"{i}%" for i in range(100, 201, 10)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode="drag"
                    )
                ], width=6, style={"paddingTop": "10px"}),
                dbc.Col([
                    dbc.Button("Rotate ↺", id="rotate-cw", style={"backgroundColor": "#3A5D85", "color": "white", "border": "none"}, className="me-2"),
                    dbc.Button("Rotate ↻", id="rotate-ccw", style={"backgroundColor": "#3A5D85", "color": "white", "border": "none"}, className="me-2")
                ], width=4, style={"paddingTop": "5px", "textAlign": "right"})
            ], className="mt-2 align-items-center"),
            dcc.Loading(
                id="loading-enhancement",
                type="default",
                children=[
                    html.Div(id='output-filename', className="text-success text-center mt-2"),
                    html.Div(id="pdf-pages-container", className="border p-3", style={"backgroundColor": "#f8f9fa", "borderRadius": "5px"})
                ]
            )
        ], width=8)
    ], className="mt-3"),

    dcc.Store(id="original-images", storage_type="memory"),
    dcc.Store(id="enhanced-images", storage_type="memory"),
    dcc.Store(id="processed-df", storage_type="memory"),
    dcc.Store(id="stored-images-base64", storage_type="memory"),
    dcc.Store(id="preview_mode", storage_type="memory"),
], fluid=True)

@app.callback(
    Output("pdf-pages-container", "children", allow_duplicate=True),
    Output("output-filename", "children"),
    Output("processed-data-container", "children"),
    Output("processing-status", "children"),
    Output("processed-df", "data"),
    Output("original-images", "data", allow_duplicate=True),
    Output("enhanced-images", "data", allow_duplicate=True),
    Output("stored-images-base64", "data", allow_duplicate=True),
    Output("preview_mode", "data"),
    Input("upload-certificate", "contents"),
    Input("enhance-certificate", "n_clicks"),
    Input("process-certificate", "n_clicks"),
    State("upload-certificate", "filename"),
    State("original-images", "data"),
    State("enhanced-images", "data"),
    prevent_initial_call=True
)
def unified_callback(contents, enhance_clicks, process_clicks, filename, original_imgs, enhanced_imgs):
    triggered_id = ctx.triggered_id

    def encode_images(images):
        img_b64s = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            img_b64s.append(img_b64)
        return img_b64s

    def display_images(b64_list):
        return [
            html.Img(
                src=f"data:image/jpeg;base64,{img_str}",
                style={"width": "100%", "borderRadius": "5px", "boxShadow": "0px 4px 6px rgba(0,0,0,0.1)"}
            ) for img_str in b64_list
        ]

    if triggered_id == "upload-certificate" and contents:
        _, content_string = contents.split(',')
        pdf_bytes = base64.b64decode(content_string)
        images = pdf2image.convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
        img_b64s = encode_images(images)
        return (
            display_images(img_b64s),
            f"Uploaded File: {filename}",
            html.Div(), dash.no_update, dash.no_update,
            img_b64s, None,
            img_b64s, "original"
        )

    elif triggered_id == "enhance-certificate" and original_imgs:
        decoded_imgs = [Image.open(io.BytesIO(base64.b64decode(img))) for img in original_imgs]
        zoom = 4.5
        enhanced_imgs = []
        for img in decoded_imgs:
            size = (int(img.width * zoom), int(img.height * zoom))
            enhanced = img.resize(size, Image.LANCZOS)
            buffer = io.BytesIO()
            enhanced.save(buffer, format="JPEG")
            enhanced_imgs.append(base64.b64encode(buffer.getvalue()).decode())
        return (
            display_images(enhanced_imgs),
            f"Enhanced File: {filename}",
            html.Div(), dash.no_update, dash.no_update,
            original_imgs, enhanced_imgs, enhanced_imgs, "enhanced"
        )

    elif triggered_id == "process-certificate" and (original_imgs or enhanced_imgs):
        active_imgs = enhanced_imgs if enhanced_imgs else original_imgs
        preview_mode = "enhanced" if enhanced_imgs else "original"

        # Save temp images for OCR processing
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for i, b64_img in enumerate(active_imgs):
                img_data = base64.b64decode(b64_img)
                img_path = os.path.join(tmpdir, f"page_{i + 1}.jpg")
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                image_paths.append(img_path)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process_all():
                executor = concurrent.futures.ThreadPoolExecutor()

                async def run_extraction():
                    return await loop.run_in_executor(executor, lambda: DataExtractor().extract_data(active_imgs))

                async def run_ocr():
                    return await loop.run_in_executor(executor, lambda: OCR_JSON().process_images(image_paths))

                data_raw, ocr_data = await asyncio.gather(run_extraction(), run_ocr())
                data = json.loads(data_raw)

                required = data.get("required", {})
                extracted = data.get("extracted", {})
                extracted_ser = pd.Series(extracted, name="Extracted Data")

                fields = {
                    "country_name": "Country",
                    "validity_in_months": "Validity",
                    "Model_number": "Model code",
                    "certificate_issue_date": "Last approval date",
                    "certificate_expiry_date": "Certification expiry date"
                }
                filtered = {v: required.get(k, "") for k, v in fields.items()}
                df_csv = pd.DataFrame([filtered])

                excel_buffer = BytesIO()
                df_csv.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_encoded = base64.b64encode(excel_buffer.getvalue()).decode()

                bbox_ocr = BoundingBoxOCR()
                extracted_df = pd.DataFrame([extracted])
                country_value = required.get("country_name", None)
                polygons = bbox_ocr.extract_box(ocr_data, extracted_df, country_value)

                # Draw bounding boxes
                images_with_boxes = []
                b64_images_with_boxes = []
                for idx, img_path in enumerate(image_paths):
                    img = cv2.imread(img_path)
                    overlay = img.copy()
                    text_positions = []
                    page_polygons = [p for p in polygons if p["page"] == idx + 1]
                    for p in page_polygons:
                        pts = np.array([[pt['x'], pt['y']] for pt in p['polygon']], dtype=np.int32).reshape((-1, 1, 2))
                        color = (255, 120, 0)
                        cv2.fillPoly(overlay, [pts], color)
                        text_positions.append((tuple(pts[0][0]), p['label']))
                    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                    font_scale = 0.7 if preview_mode != "enhanced" else 0.7 * 4.5
                    font_thickness = 2 if preview_mode != "enhanced" else int(2 * 4.5)
                    for pos, label in text_positions:
                        cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (139, 0, 0), font_thickness)
                    _, buffer = cv2.imencode(".jpg", img)
                    img_b64 = base64.b64encode(buffer).decode()
                    b64_images_with_boxes.append(img_b64)
                    images_with_boxes.append(html.Img(src=f"data:image/jpeg;base64,{img_b64}", style={"width": "100%"}))

                return (
                    images_with_boxes,
                    dash.no_update,
                    html.Div([
                    html.H5("Required Format", className="text-primary mt-2"),
                    html.Div([
                        html.Div([
                            html.Div(f"{k}:", style={
                                "fontFamily": "monospace",
                                "fontWeight": "bold",
                                "width": "200px",
                                "textAlign": "right",
                                "paddingRight": "15px"
                            }),
                            dcc.Input(
                                id={'type': 'required-input', 'index': k},
                                value=str(v),
                                type="text",
                                debounce=True,
                                style={
                                    "fontFamily": "monospace",
                                    "flex": "1",
                                    "padding": "6px 10px",
                                    "border": "1px solid #ced4da",
                                    "borderRadius": "4px"
                                },
                                className="mb-2"
                            )
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "8px"
                        }) for k, v in filtered.items()
                    ], style={
                        "backgroundColor": "#f8f9fa",
                        "border": "1px solid #ced4da",
                        "borderRadius": "5px",
                        "padding": "15px",
                        "fontFamily": "monospace",
                        "maxWidth": "600px"
                    }),
                    html.Button("Download Excel", id="download-btn", className="btn btn-success mt-3"),
                    dcc.Download(id="download-required-excel"),

                    html.H5("Actual Data", className="text-primary mt-4"),
                    html.Pre(
                        extracted_ser.to_string(),
                        className="border p-3",
                        style={"fontFamily": "monospace", "backgroundColor": "#f8f9fa"}
                    )
                ]),
                f"Processed {preview_mode} certificate",
                df_csv.to_dict("records"),
                original_imgs,
                enhanced_imgs,
                b64_images_with_boxes, preview_mode
                )

            result = loop.run_until_complete(process_all())
            loop.close()
            return result

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("download-required-excel", "data"),
    Input("download-btn", "n_clicks"),
    State({'type': 'required-input', 'index': ALL}, 'value'),
    State({'type': 'required-input', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def download_required_data(n_clicks, values, ids):
    # Map values back to field names
    edited_data = {item['index']: val for item, val in zip(ids, values)}
    
    # Create DataFrame and Excel buffer
    df = pd.DataFrame([edited_data])
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    return dcc.send_bytes(buffer.read(), "required_data.xlsx")

@app.callback(
    Output("original-images", "data", allow_duplicate=True),
    Output("enhanced-images", "data", allow_duplicate=True),
    Output("pdf-pages-container", "children", allow_duplicate=True),
    Input("rotate-cw", "n_clicks"),
    Input("rotate-ccw", "n_clicks"),
    State("original-images", "data"),
    State("enhanced-images", "data"),
    prevent_initial_call=True
)
def rotate_certificate(cw_clicks, ccw_clicks, orig_imgs, enh_imgs):
    triggered = ctx.triggered_id
    angle = -90 if triggered == "rotate-ccw" else 90
    active_imgs = enh_imgs or orig_imgs
    if not active_imgs:
        return dash.no_update, dash.no_update, dash.no_update

    rotated = []
    for b64_img in active_imgs:
        img = Image.open(BytesIO(base64.b64decode(b64_img)))
        rotated_img = img.rotate(angle, expand=True)
        buffer = BytesIO()
        rotated_img.save(buffer, format="JPEG")
        rotated.append(base64.b64encode(buffer.getvalue()).decode())

    return (
        rotated if not enh_imgs else None,
        rotated if enh_imgs else None,
        [html.Img(src=f"data:image/jpeg;base64,{img}", style={"width": "100%"}) for img in rotated]
    )

@app.callback(
    Output("pdf-pages-container", "children", allow_duplicate=True),
    Input("zoom-slider", "value"),
    Input("preview_mode", "data"),
    State("stored-images-base64", "data"),
    prevent_initial_call=True,
)
def update_preview_zoom(zoom_value, preview_mode, base64_images):
    if not base64_images:
        return dash.no_update
    if(preview_mode=="original"):
        fact = 0.75
    else:
        fact = 0.167
    scale_factor = (zoom_value / 100.0)*fact

    zoomed_images = [
        html.Div([
            html.Img(
                src=f"data:image/jpeg;base64,{b64_img}",
                style={
                    "transform": f"scale({scale_factor})",
                    "transformOrigin": "top left",
                    "display": "block",
                    "width": "fit_content"
                }
            )
        ], style={"overflow": "auto"})  
        for b64_img in base64_images
    ]

    return html.Div(
        zoomed_images,
        style={
            "overflow": "auto",
            "maxHeight": "85vh",
            "whiteSpace": "nowrap"
        }
    )

if __name__ == "__main__":
    app.run(debug=True, port=8777)
    
    
