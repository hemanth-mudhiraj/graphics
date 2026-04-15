"""FastAPI backend for CheXNet chest X-ray classification."""

import os
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from model import load_model, predict

# ── Configuration ──
MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).parent.parent / "best_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Global model reference ──
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Set MODEL_PATH env var or place best_model.pth in the project root.")
        print("The app will start but predictions will fail until the model is available.")
    else:
        print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
        model = load_model(MODEL_PATH, DEVICE)
        print("Model loaded successfully!")
    yield


app = FastAPI(title="CheXNet - Chest X-ray Classifier", lifespan=lifespan)

# ── Static files & templates ──
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place best_model.pth in the project root.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (JPEG, PNG, etc.)")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    try:
        results = predict(model, image_bytes, DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {"filename": file.filename, "predictions": results}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
