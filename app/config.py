import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    APP_NAME: str = "T-Shirt Mockup Generator API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    API_PREFIX: str = "/api/v1"
    
    # File storage settings
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "output"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"

settings = Settings() 