from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend import schemas
import model_load

app = FastAPI(title="Plant Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Plant Disease Detection API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/analyze", response_model=schemas.AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        image_bytes = await file.read()
        predictions_data = model_load.predict(image_bytes)
        remedies_data = model_load.get_remedies([p["label"] for p in predictions_data])

        # Convert dictionaries to Pydantic models
        predictions = [schemas.Prediction(**p) for p in predictions_data]
        remedies = [schemas.Remedy(**r) for r in remedies_data]

        return schemas.AnalyzeResponse(
            filename=file.filename,
            predictions=predictions,
            remedies=remedies,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
