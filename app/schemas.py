# app/schemas.py
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=10, max_items=10)

class PredictResponse(BaseModel):
    prediction: List[float]
