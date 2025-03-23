import cv2
import numpy as np
import os
from fastapi import UploadFile
import aiofiles
from typing import Dict, Tuple, Any

from app.interfaces.mockup_generator import MockupGenerator

class MockupService:
    """Service for handling mockup generation requests"""
    
    def __init__(self, mockup_generator: MockupGenerator):
        self.mockup_generator = mockup_generator
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
    
    async def _save_upload_file(self, upload_file: UploadFile, file_path: str) -> str:
        """Save an uploaded file to disk"""
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
        return file_path
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
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
            print(f"Warning: Aspect ratio mismatch when resizing image. " 
                  f"Source: {current_ratio:.2f}, Target: {target_ratio:.2f}")
        
        # Determine interpolation method based on whether we're upscaling or downscaling
        interpolation = cv2.INTER_CUBIC if (target_w > w or target_h > h) else cv2.INTER_AREA
        
        # Resize the image
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    
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
        
        # Save uploaded files
        source_image_path = await self._save_upload_file(
            source_image, f"uploads/{mockup_id}_source.jpg")
        source_mask_path = await self._save_upload_file(
            source_mask, f"uploads/{mockup_id}_mask.png")
        source_depth_path = await self._save_upload_file(
            source_depth, f"uploads/{mockup_id}_depth.png")
        design_image_path = await self._save_upload_file(
            design_image, f"uploads/{mockup_id}_design.png")
        
        # Load images
        source_img = cv2.imread(source_image_path)
        # Load mask with -1 flag to preserve all channels including alpha if present
        source_mask_img_original = cv2.imread(source_mask_path, cv2.IMREAD_UNCHANGED)
        source_depth_img = cv2.imread(source_depth_path, cv2.IMREAD_UNCHANGED)
        design_img = cv2.imread(design_image_path, cv2.IMREAD_UNCHANGED)
        
        # Validate images
        if source_img is None:
            raise ValueError("Failed to load source image")
        if source_mask_img_original is None:
            raise ValueError("Failed to load source mask")
        if source_depth_img is None:
            raise ValueError("Failed to load depth map")
        if design_img is None:
            raise ValueError("Failed to load design image")
        
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
        
        # Check if design has alpha channel, add one if not
        if len(design_img.shape) == 3 and design_img.shape[2] == 3:
            # Convert BGR to BGRA by adding an alpha channel
            alpha = np.ones((design_img.shape[0], design_img.shape[1]), dtype=design_img.dtype) * 255
            design_img = cv2.merge((design_img[:, :, 0], design_img[:, :, 1], design_img[:, :, 2], alpha))
        elif len(design_img.shape) != 3 or design_img.shape[2] != 4:
            raise ValueError("Design image must be a color image with an alpha channel")
        
        # Convert hex color to RGB
        rgb_color = self._hex_to_rgb(color_code)
        
        # Generate mockup
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
        
        # Save the result
        output_path = f"output/{mockup_id}.jpg"
        cv2.imwrite(output_path, result_image)
        
        # Clean up temporary files
        for path in [source_image_path, source_mask_path, source_depth_path, design_image_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return {
            "file_path": output_path,
            "width": result_image.shape[1],
            "height": result_image.shape[0]
        } 