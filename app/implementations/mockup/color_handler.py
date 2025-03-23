import cv2
import numpy as np
from typing import Tuple, Dict, Any
from app.utils.logging_config import get_logger
from app.utils.debug_utils import debug_function_args, debug_exception, debug_timing

class ColorHandler:
    """Handles color-related operations for the mockup generator"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @debug_function_args
    def determine_color_mode(
        self, 
        color_code: Tuple[int, int, int], 
        color_mode: str
    ) -> str:
        """Determine if color is light or dark based on perceived brightness."""
        if color_mode != 'auto':
            return color_mode
            
        # Calculate perceived brightness (ITU-R BT.709)
        brightness = 0.2126 * color_code[0] + 0.7152 * color_code[1] + 0.0722 * color_code[2]
        determined_mode = 'light' if brightness > 128 else 'dark'
        self.logger.debug(f"Auto color mode determined: {determined_mode} (brightness: {brightness:.1f})")
        return determined_mode
    
    @debug_timing
    @debug_exception
    def apply_tshirt_color(
        self, 
        source_image: np.ndarray, 
        source_mask: np.ndarray, 
        color_code: Tuple[int, int, int], 
        color_mode: str
    ) -> np.ndarray:
        """Apply color to the t-shirt area using appropriate blending method."""
        # Create a copy of the source image
        colored_tshirt_image = source_image.copy()
        
        # Determine color application method
        color_mode = self.determine_color_mode(color_code, color_mode)
        
        # Extract mask
        mask = source_mask > 0
        
        if color_mode == 'light':
            colored_tshirt_image = self._apply_light_color(source_image, mask, color_code)
        else:
            colored_tshirt_image = self._apply_dark_color(source_image, mask, color_code)
            
        return colored_tshirt_image
    
    @debug_exception
    def _apply_light_color(
        self, 
        source_image: np.ndarray, 
        mask: np.ndarray, 
        color_code: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply light color using HSV color space to preserve shading."""
        self.logger.debug("Applying light color using HSV method")
        # Convert RGB color_code to HSV
        color_rgb = np.array([[color_code]], dtype=np.uint8)
        color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        
        # Convert source image to HSV and apply color
        hsv_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
        hsv_source[mask, 0] = color_hsv[0]  # Hue
        hsv_source[mask, 1] = color_hsv[1]  # Saturation
        
        return cv2.cvtColor(hsv_source, cv2.COLOR_HSV2BGR)
    
    @debug_exception
    def _apply_dark_color(
        self, 
        source_image: np.ndarray, 
        mask: np.ndarray, 
        color_code: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply dark color using multiply blending for better results."""
        self.logger.debug("Applying dark color using multiply blending")
        # Create a copy of the source image
        colored_image = source_image.copy()
        
        # Create a solid color image with the target color
        # Get the number of channels from the source image
        num_channels = source_image.shape[2]
        solid_color = np.zeros_like(source_image)
        
        # Handle different channel counts
        if num_channels == 3:
            solid_color[:] = (color_code[2], color_code[1], color_code[0])  # BGR format
        elif num_channels == 4:
            solid_color[:] = (color_code[2], color_code[1], color_code[0], 255)  # BGRA format
        else:
            self.logger.warning(f"Unexpected number of channels: {num_channels}")
            # Fallback for other channel counts
            for c in range(min(3, num_channels)):
                solid_color[:, :, c] = color_code[2-c]  # Reverse RGB to BGR
        
        # Extract lighting information from the original t-shirt
        luminance = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        
        # Apply color with lighting preservation
        for c in range(min(3, num_channels)):  # Only apply to color channels (BGR)
            # Multiply mode blending: preserves shadows and highlights
            colored_image[mask, c] = np.clip(
                (solid_color[mask, c].astype(float) * luminance[mask] * 1.5),
                0,
                255
            ).astype(np.uint8)
            
        return colored_image 