from typing import List
from pydantic import BaseModel, Field

class Prediction(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class Remedy(BaseModel):
    label: str = Field(..., description="Disease or condition label the remedy applies to")
    actions: List[str] = Field(default_factory=list, description="Actionable treatment or prevention steps")

class AnalyzeResponse(BaseModel):
    filename: str
    predictions: List[Prediction]
    remedies: List[Remedy]
