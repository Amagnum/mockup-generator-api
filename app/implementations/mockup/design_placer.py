import numpy as np
from typing import Dict, Tuple
from app.utils.logging_config import get_logger
from app.utils.debug_utils import debug_exception

class DesignPlacer:
    """Handles design placement calculations for the mockup generator"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @debug_exception
    def compute_design_placement(
        self, 
        image_shape: Tuple[int, int], 
        design_shape: Tuple[int, int], 
        location: Tuple[int, int], 
        scale_factor: float
    ) -> Dict[str, int]:
        """Compute the bounding box for design placement."""
        image_height, image_width = image_shape
        dh, dw = design_shape
        
        scaled_dw = int(dw * scale_factor)
        scaled_dh = int(dh * scale_factor)
        x, y = location
        
        x_min = int(x - scaled_dw / 2)
        x_max = x_min + scaled_dw
        y_min = int(y - scaled_dh / 2)
        y_max = y_min + scaled_dh
        
        # Check if design placement is valid
        if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            self.logger.warning(f"Design placement exceeds image boundaries: "
                              f"x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, "
                              f"image_width={image_width}, image_height={image_height}")
            if x_min < 0:
                self.logger.warning(f"Adjusting x_min from {x_min} to 0")
                x_min = 0
            if y_min < 0:
                self.logger.warning(f"Adjusting y_min from {y_min} to 0")
                y_min = 0
            if x_max > image_width:
                self.logger.warning(f"Adjusting x_max from {x_max} to {image_width}")
                x_max = image_width
            if y_max > image_height:
                self.logger.warning(f"Adjusting y_max from {y_max} to {image_height}")
                y_max = image_height
            
        return {
            'x_min': x_min, 
            'x_max': x_max, 
            'y_min': y_min, 
            'y_max': y_max,
            'scaled_dw': scaled_dw,
            'scaled_dh': scaled_dh,
            'original_dw': dw,
            'original_dh': dh
        } 