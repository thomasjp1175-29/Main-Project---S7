"""
Fixture Design API - Milestone 7 + 8 backend

Wraps standalone_feature_analyzer.analyze_step() in a web endpoint so a
browser can upload a .stp file and get back the feature report,
3-2-1 support points (now stress + CoG aware), AND a .glb 3D mesh to
display in a viewer.

Install:
    pip install fastapi uvicorn python-multipart cadquery trimesh

Run (from this folder):
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs to test /analyze directly, or
open index.html (Milestone 8 frontend) in a browser for the full
upload + 3D viewer experience.
"""

import os
import uuid
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from standalone_feature_analyzer import (
    analyze_step,
    MATERIAL_YIELD_MPA,
    DEFAULT_MATERIAL,
    DEFAULT_SAFETY_FACTOR,
    DEFAULT_APPLIED_FORCE_N,
    DEFAULT_PIN_CONTACT_RADIUS_MM,
)

app = FastAPI(title="Fixture Design API")

# Allows index.html (opened directly as a file, or served from a
# different port) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where generated .glb files get saved and served from, e.g.
# http://127.0.0.1:8000/models/<some-id>.glb
MODELS_DIR = os.path.join(os.path.dirname(__file__), "generated_models")
os.makedirs(MODELS_DIR, exist_ok=True)
app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fixture Design API is running"}


@app.get("/materials")
def list_materials():
    """So the frontend can populate a material dropdown without
    hardcoding the yield-strength table twice."""
    return {"materials": MATERIAL_YIELD_MPA, "default": DEFAULT_MATERIAL}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    material: str = Form(DEFAULT_MATERIAL),
    applied_force_n: float = Form(DEFAULT_APPLIED_FORCE_N),
    pin_radius_mm: float = Form(DEFAULT_PIN_CONTACT_RADIUS_MM),
    safety_factor: float = Form(DEFAULT_SAFETY_FACTOR),
):
    if not file.filename.lower().endswith((".stp", ".step")):
        raise HTTPException(status_code=400, detail="Please upload a .stp or .step file")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    glb_filename = f"{uuid.uuid4().hex}.glb"
    glb_path = os.path.join(MODELS_DIR, glb_filename)

    try:
        result = analyze_step(
            tmp_path,
            glb_out_path=glb_path,
            material=material,
            applied_force_n=applied_force_n,
            pin_radius_mm=pin_radius_mm,
            safety_factor=safety_factor,
        )
        return {
            "filename": file.filename,
            "through_holes": result["through_holes"],
            "corner_fillets": result["corner_fillets"],
            "unknown_cylinders": result["unknown_cylinders"],
            "center_of_mass": result["center_of_mass"],
            "center_of_mass_gltf": result["center_of_mass_gltf"],
            "support_points": result["support_points"],
            "support_points_gltf": result["support_points_gltf"],
            "clamp_analysis": result["clamp_analysis"],
            "report_text": result["report_text"],
            "glb_url": f"/models/{glb_filename}",
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        os.unlink(tmp_path)