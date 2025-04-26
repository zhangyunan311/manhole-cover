from pydantic import BaseModel
from typing import List, Optional

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class ImageDetectionResponse(BaseModel):
    detections: List[DetectionResult]
    result_image: str

class DetectionSettings(BaseModel):
    confidence: Optional[float] = None
    iou: Optional[float] = None
    device: Optional[str] = None
    half: Optional[bool] = None
