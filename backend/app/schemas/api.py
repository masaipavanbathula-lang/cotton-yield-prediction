from typing import Any, Optional
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    Growth_Cycle: int = Field(..., ge=1, le=365)
    Sunlight_Hours: float = Field(..., ge=0, le=24)
    Precipitation: float = Field(..., ge=0)
    Average_Temperature: float = Field(..., ge=-30, le=60)
    Drought_Days: int = Field(..., ge=0, le=365)
    Flood_Days: int = Field(..., ge=0, le=365)
    Soil_Moisture: float = Field(..., ge=0, le=100)
    Soil_pH: float = Field(..., ge=0, le=14)
    CO2_Concentration: float = Field(..., ge=0)
    Sunlight_Level: str
    Flood_Level: Optional[str] = None
    Drought_Level: Optional[str] = None
    model_name: Optional[str] = None
class PredictionResponse(BaseModel):
    predicted_yield: float
    model_name: str
    model_version: str
    status: str
    input_features: list[str]
class ApiResponse(BaseModel):
    status: str = "success"
    data: Any
    meta: dict[str, Any] = {}
