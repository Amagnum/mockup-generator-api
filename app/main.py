from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uuid
import os
import logging
import time
from typing import Any, Dict, Optional

from app.services.mockup_service import MockupService
from app.dependencies import get_mockup_service
from app.schemas.mockup_schemas import MockupResponse, ErrorResponse
from app.utils.logging_config import setup_logging, get_logger
from app.config import settings

# Setup logging
logger = setup_logging(
    log_level=logging.DEBUG if settings.DEBUG else logging.INFO
)

# Create the FastAPI application
app = FastAPI(
    title="T-Shirt Mockup Generator API",
    description="API for generating realistic t-shirt mockups with custom designs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.4f}s: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Ensure output directory exists
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.debug("Health check endpoint called")
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
    color_method: str = Form("standard"),
    session_id: Optional[str] = Form(None),
    color_config: Optional[str] = Form(None),
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
    - **color_method**: Coloring algorithm: 'standard', 'intrinsic', 'lum-v2', or 'int-v2' (default: standard)
    - **session_id**: Optional session ID to update the same image across requests
    - **color_config**: JSON string with parameters to control the recoloring process (used with 'int-v2' method):
        - **large_scale_blur**: Size of kernel for large-scale shading (default: 21)
          - Larger values (e.g., 31) create smoother, more uniform shading
          - Smaller values (e.g., 15) preserve more of the original shading variations
        
        - **medium_scale_blur**: Size of kernel for medium-scale details (default: 7)
          - Larger values capture broader fabric patterns
          - Smaller values focus on finer fabric weave patterns
        
        - **fine_scale_blur**: Size of kernel for fine details (default: 3)
          - Smaller values preserve more micro-texture details
          - Increasing this slightly can reduce noise in the final result
        
        - **large_scale_weight**: Weight for large-scale shading (default: 0.7)
          - Higher values (e.g., 1.0) create more pronounced overall shading
          - Lower values (e.g., 0.5) flatten the overall shading
        
        - **medium_scale_weight**: Weight for medium details (default: 1.5)
          - Higher values enhance fabric patterns and folds
          - Lower values create a smoother appearance
        
        - **fine_scale_weight**: Weight for fine details (default: 2.0)
          - Higher values (e.g., 3.0) create more textured, fabric-like appearance
          - Lower values (e.g., 1.0) create a smoother, less detailed result
        
        - **min_shading**: Minimum shading value to prevent black areas (default: 0.05)
          - Higher values (e.g., 0.2) prevent very dark shadows, creating a flatter look
          - Lower values (e.g., 0.01) allow deeper shadows for more contrast
        
        - **shading_boost**: Boost factor for final shading (default: 1.2)
          - Higher values increase overall contrast
          - Lower values create a flatter, more uniform coloring
        
        - **base_detail_preservation**: Base level of detail to preserve (default: 0.15)
          - Higher values (e.g., 0.3) retain more of the original fabric appearance
          - Lower values (e.g., 0.05) apply more of the target color
        
        - **texture_detail_weight**: How much additional detail to preserve in textured areas (default: 0.25)
          - Higher values preserve more detail in complex fabric areas
          - Lower values create more uniform coloring across the fabric
        
        - **saturation_influence**: How much color saturation reduces detail preservation (default: 0.3)
          - Higher values make saturated colors appear more uniform
          - Lower values preserve more texture even with saturated colors
    
    Example color_config for cotton t-shirts:
    ```json
    {
        "large_scale_weight": 0.7,
        "medium_scale_weight": 1.5,
        "fine_scale_weight": 2.0,
        "base_detail_preservation": 0.15,
        "texture_detail_weight": 0.25
    }
    ```
    
    Example color_config for textured fabrics like linen:
    ```json
    {
        "large_scale_weight": 0.6,
        "medium_scale_weight": 1.8,
        "fine_scale_weight": 2.5,
        "base_detail_preservation": 0.2,
        "texture_detail_weight": 0.35
    }
    ```
    
    Example color_config for smooth fabrics like silk:
    ```json
    {
        "large_scale_weight": 0.8,
        "medium_scale_weight": 1.2,
        "fine_scale_weight": 1.5,
        "base_detail_preservation": 0.1,
        "texture_detail_weight": 0.15
    }
    ```
    """
    req_logger = get_logger(__name__)
    try:
        # Parse color_config from JSON string if provided
        parsed_color_config = None
        if color_config:
            try:
                import json
                parsed_color_config = json.loads(color_config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid color_config JSON: {str(e)}")

        # Use provided session_id or generate a new unique ID for this mockup
        mockup_id = session_id if session_id else str(uuid.uuid4())
        req_logger.info(f"Starting mockup generation with ID: {mockup_id}")
        
        # Log input parameters
        req_logger.debug(f"Parameters: color={color_code}, location=({location_x},{location_y}), "
                        f"scale={scale_factor}, shading={shading_strength}, mode={color_mode}, "
                        f"method={color_method}, config={parsed_color_config}")
        
        # Process the mockup with parsed color_config
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
            color_mode=color_mode,
            color_method=color_method,
            color_config=parsed_color_config
        )
        
        req_logger.info(f"Mockup generation completed successfully: {mockup_id}")
        return {
            "mockup_id": mockup_id,
            "file_path": result["file_path"],
            "download_url": f"/mockups/{mockup_id}/download"
        }
        
    except ValueError as e:
        req_logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        req_logger.error(f"Error generating mockup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating mockup: {str(e)}")

@app.get(
    "/mockups/{mockup_id}/download",
    responses={404: {"model": ErrorResponse}}
)
async def download_mockup(mockup_id: str):
    """Download a generated mockup by ID"""
    req_logger = get_logger(__name__)
    file_path = f"output/{mockup_id}.png"
    
    req_logger.debug(f"Download request for mockup ID: {mockup_id}")
    
    if not os.path.exists(file_path):
        req_logger.warning(f"Mockup not found: {mockup_id}")
        raise HTTPException(status_code=404, detail="Mockup not found")
    
    req_logger.info(f"Serving mockup file: {file_path}")
    return FileResponse(
        file_path, 
        media_type="image/png",
        filename=f"tshirt_mockup_{mockup_id}.png"
    )

@app.delete(
    "/mockups/{mockup_id}",
    responses={404: {"model": ErrorResponse}}
)
async def delete_mockup(mockup_id: str):
    """Delete a generated mockup by ID"""
    req_logger = get_logger(__name__)
    file_path = f"output/{mockup_id}.png"
    
    req_logger.debug(f"Delete request for mockup ID: {mockup_id}")
    
    if not os.path.exists(file_path):
        req_logger.warning(f"Mockup not found for deletion: {mockup_id}")
        raise HTTPException(status_code=404, detail="Mockup not found")
    
    os.remove(file_path)
    req_logger.info(f"Mockup deleted successfully: {mockup_id}")
    return {"status": "success", "message": f"Mockup {mockup_id} deleted successfully"} 