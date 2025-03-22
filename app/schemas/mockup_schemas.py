from pydantic import BaseModel, Field
from typing import Optional, Tuple

class MockupResponse(BaseModel):
    """Response schema for a generated mockup"""
    mockup_id: str = Field(..., description="Unique identifier for the mockup")
    file_path: str = Field(..., description="Server file path of the generated mockup")
    download_url: str = Field(..., description="URL to download the mockup")

class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str = Field(..., description="Error details") 