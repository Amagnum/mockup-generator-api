from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import os
from typing import Optional

from app.services.mockup_service import MockupService
from app.dependencies import get_mockup_service
from app.schemas.mockup_schemas import MockupResponse, ErrorResponse

# Create the FastAPI application
app = FastAPI(
    title="T-Shirt Mockup Generator API",
    description="API for generating realistic t-shirt mockups with custom designs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "T-Shirt Mockup Generator API is running"}

@app.post(
    "/mockups/generate", 
    response_model=MockupResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def generate_mockup(
    source_image: UploadFile = File(...),
    source_mask: UploadFile = File(...),
    source_depth: UploadFile = File(...),
    design_image: UploadFile = File(...),
    color_code: str = Form(...),
    location_x: int = Form(...),
    location_y: int = Form(...),
    scale_factor: float = Form(0.5),
    shading_strength: float = Form(0.7),
    color_mode: str = Form("auto"),
    mockup_service: MockupService = Depends(get_mockup_service)
):
    """
    Generate a t-shirt mockup with the provided images and parameters
    
    - **source_image**: Person wearing a white t-shirt
    - **source_mask**: Binary mask of the t-shirt area
    - **source_depth**: 16-bit depth map of the source image
    - **design_image**: Design to apply (with alpha channel)
    - **color_code**: Hex color code for the t-shirt (e.g., #FF0000)
    - **location_x**: X coordinate for design placement
    - **location_y**: Y coordinate for design placement
    - **scale_factor**: Scale factor for the design (default: 0.5)
    - **shading_strength**: Strength of shading effect (0-1, default: 0.7)
    - **color_mode**: Color application mode: 'auto', 'light', or 'dark' (default: auto)
    """
    try:
        # Generate a unique ID for this mockup
        mockup_id = str(uuid.uuid4())
        
        # Process the mockup
        result = await mockup_service.generate_mockup(
            mockup_id=mockup_id,
            source_image=source_image,
            source_mask=source_mask,
            source_depth=source_depth,
            design_image=design_image,
            color_code=color_code,
            location=(location_x, location_y),
            scale_factor=scale_factor,
            shading_strength=shading_strength,
            color_mode=color_mode
        )
        
        return {
            "mockup_id": mockup_id,
            "file_path": result["file_path"],
            "download_url": f"/mockups/{mockup_id}/download"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating mockup: {str(e)}")

@app.get(
    "/mockups/{mockup_id}/download",
    responses={404: {"model": ErrorResponse}}
)
async def download_mockup(mockup_id: str):
    """Download a generated mockup by ID"""
    file_path = f"output/{mockup_id}.jpg"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Mockup not found")
    
    return FileResponse(
        file_path, 
        media_type="image/jpeg",
        filename=f"tshirt_mockup_{mockup_id}.jpg"
    )

@app.delete(
    "/mockups/{mockup_id}",
    responses={404: {"model": ErrorResponse}}
)
async def delete_mockup(mockup_id: str):
    """Delete a generated mockup by ID"""
    file_path = f"output/{mockup_id}.jpg"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Mockup not found")
    
    os.remove(file_path)
    return {"status": "success", "message": f"Mockup {mockup_id} deleted successfully"} 