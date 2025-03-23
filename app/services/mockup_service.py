import cv2
import numpy as np
import os
from fastapi import UploadFile
import aiofiles
from typing import Dict, Tuple, Any

from app.interfaces.mockup_generator import MockupGenerator
from app.utils.logging_config import get_logger
from app.utils.debug_utils import save_debug_image, debug_timing, debug_exception, debug_function_args

class MockupService:
    """Service for handling mockup generation requests"""
    
    def __init__(self, mockup_generator: MockupGenerator):
        self.mockup_generator = mockup_generator
        self.logger = get_logger(__name__)
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
    
    @debug_exception
    async def _save_upload_file(self, upload_file: UploadFile, file_path: str) -> str:
        """Save an uploaded file to disk"""
        self.logger.debug(f"Saving uploaded file to {file_path}")
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
        return file_path
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @debug_exception
    def _resize_to_match(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize image to match target dimensions while preserving aspect ratio if possible"""
        if image.shape[:2] == target_shape[:2]:
            return image
            
        # Get current dimensions
        h, w = image.shape[:2]
        target_h, target_w = target_shape[:2]
        
        # Check if aspect ratios are close enough (within 1%)
        current_ratio = w / h
        target_ratio = target_w / target_h
        ratio_difference = abs(current_ratio - target_ratio) / target_ratio
        
        if ratio_difference > 0.01:
            self.logger.warning(f"Aspect ratio mismatch when resizing image. " 
                  f"Source: {current_ratio:.2f}, Target: {target_ratio:.2f}")
        
        # Determine interpolation method based on whether we're upscaling or downscaling
        interpolation = cv2.INTER_CUBIC if (target_w > w or target_h > h) else cv2.INTER_AREA
        
        # Resize the image
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        self.logger.debug(f"Resized image from {(w, h)} to {(target_w, target_h)}")
        return resized
    
    @debug_timing
    @debug_exception
    @debug_function_args
    async def generate_mockup(
        self,
        mockup_id: str,
        source_image: UploadFile,
        source_mask: UploadFile,
        source_depth: UploadFile,
        design_image: UploadFile,
        color_code: str,
        location: Tuple[int, int],
        scale_factor: float,
        shading_strength: float = 0.7,
        color_mode: str = 'auto'
    ) -> Dict[str, Any]:
        """Process uploaded files and generate a mockup"""
        self.logger.info(f"Starting mockup generation for ID: {mockup_id}")
        
        # Save uploaded files
        source_image_path = await self._save_upload_file(
            source_image, f"uploads/{mockup_id}_source.png")
        source_mask_path = await self._save_upload_file(
            source_mask, f"uploads/{mockup_id}_mask.png")
        source_depth_path = await self._save_upload_file(
            source_depth, f"uploads/{mockup_id}_depth.png")
        design_image_path = await self._save_upload_file(
            design_image, f"uploads/{mockup_id}_design.png")
        
        # Load images with full quality
        source_img = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)
        source_mask_img_original = cv2.imread(source_mask_path, cv2.IMREAD_UNCHANGED)
        source_depth_img = cv2.imread(source_depth_path, cv2.IMREAD_UNCHANGED)
        design_img = cv2.imread(design_image_path, cv2.IMREAD_UNCHANGED)
        
        # Save debug images
        save_debug_image(source_img, f"{mockup_id}_01_source", mockup_id)
        save_debug_image(source_mask_img_original, f"{mockup_id}_02_mask_original", mockup_id)
        save_debug_image(source_depth_img, f"{mockup_id}_03_depth", mockup_id)
        save_debug_image(design_img, f"{mockup_id}_04_design", mockup_id)
        
        # Validate images
        if source_img is None:
            self.logger.error(f"Failed to load source image: {source_image_path}")
            raise ValueError("Failed to load source image")
        if source_mask_img_original is None:
            self.logger.error(f"Failed to load source mask: {source_mask_path}")
            raise ValueError("Failed to load source mask")
        if source_depth_img is None:
            self.logger.error(f"Failed to load depth map: {source_depth_path}")
            raise ValueError("Failed to load depth map")
        if design_img is None:
            self.logger.error(f"Failed to load design image: {design_image_path}")
            raise ValueError("Failed to load design image")
        
        # Log image properties
        self.logger.debug(f"Source image: shape={source_img.shape}, dtype={source_img.dtype}")
        self.logger.debug(f"Source mask: shape={source_mask_img_original.shape}, dtype={source_mask_img_original.dtype}")
        self.logger.debug(f"Source depth: shape={source_depth_img.shape}, dtype={source_depth_img.dtype}")
        self.logger.debug(f"Design image: shape={design_img.shape}, dtype={design_img.dtype}")
        
        # Resize mask and depth to match source image dimensions if needed
        source_mask_img_original = self._resize_to_match(source_mask_img_original, source_img.shape)
        source_depth_img = self._resize_to_match(source_depth_img, source_img.shape)
        
        # Process the mask to ensure it's a proper binary black and white image
        if len(source_mask_img_original.shape) == 3:
            # If mask has multiple channels (RGB or RGBA)
            if source_mask_img_original.shape[2] == 4:
                # Convert RGB to grayscale, ignoring alpha
                source_mask_img = cv2.cvtColor(source_mask_img_original[:, :, :3], cv2.COLOR_BGR2GRAY)
                # Get alpha channel and create mask where alpha is 0 (transparent)
                alpha_mask = source_mask_img_original[:, :, 3] == 0
                # Set transparent areas to black (0) in the grayscale image
                source_mask_img[alpha_mask] = 0
            else:
                # Convert RGB to grayscale
                source_mask_img = cv2.cvtColor(source_mask_img_original, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            source_mask_img = source_mask_img_original
        
        # Ensure the mask is binary (0 or 255)
        _, source_mask_img = cv2.threshold(source_mask_img, 127, 255, cv2.THRESH_BINARY)
        save_debug_image(source_mask_img, f"{mockup_id}_05_mask_processed", mockup_id)
        
        # Check if design has alpha channel, add one if not
        if len(design_img.shape) == 3 and design_img.shape[2] == 3:
            # Convert BGR to BGRA by adding an alpha channel
            self.logger.debug("Adding alpha channel to design image")
            alpha = np.ones((design_img.shape[0], design_img.shape[1]), dtype=design_img.dtype) * 255
            design_img = cv2.merge((design_img[:, :, 0], design_img[:, :, 1], design_img[:, :, 2], alpha))
            save_debug_image(design_img, f"{mockup_id}_06_design_with_alpha", mockup_id)
        elif len(design_img.shape) != 3 or design_img.shape[2] != 4:
            self.logger.error(f"Invalid design image format: shape={design_img.shape}")
            raise ValueError("Design image must be a color image with an alpha channel")
        
        # Convert hex color to RGB
        rgb_color = self._hex_to_rgb(color_code)
        self.logger.debug(f"T-shirt color: hex={color_code}, rgb={rgb_color}")
        
        # Generate mockup
        try:
            result_image = self.mockup_generator.generate_tshirt_mockup(
                source_image=source_img,
                source_mask=source_mask_img,
                source_depth=source_depth_img,
                design_image=design_img,
                color_code=rgb_color,
                location=location,
                scale_factor=scale_factor,
                shading_strength=shading_strength,
                color_mode=color_mode
            )
            save_debug_image(result_image, f"{mockup_id}_07_final_result", mockup_id)
        except Exception as e:
            self.logger.error(f"Error in mockup generation: {str(e)}", exc_info=True)
            raise
        
        # Save the result as PNG with maximum quality
        output_path = f"output/{mockup_id}.png"
        cv2.imwrite(output_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = no compression for best quality
        self.logger.info(f"Mockup saved to {output_path}")
        
        # Clean up temporary files
        for path in [source_image_path, source_mask_path, source_depth_path, design_image_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return {
            "file_path": output_path,
            "width": result_image.shape[1],
            "height": result_image.shape[0]
        } 