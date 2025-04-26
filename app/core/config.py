from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "YOLO Detection API"
    API_V1_STR: str = "/api/v1"

    # YOLO Model settings
    MODEL_PATH: str = "best.pt"  # 默认使用YOLOv8n模型
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    DEVICE: str = "cpu"  # 默认使用 CPU
    HALF: bool = False  # Use half precision

    # Upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    class Config:
        case_sensitive = True

settings = Settings()
