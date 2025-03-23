import cv2
import numpy as np
from typing import Dict, Tuple
from app.utils.logging_config import get_logger
from app.utils.debug_utils import debug_timing, debug_exception
from app.utils.debug_utils import save_debug_image

class DesignWarper:
    """Handles design warping based on depth maps for the mockup generator"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @debug_timing
    @debug_exception
    def process_depth_map(
        self, 
        depth_map: np.ndarray, 
        design_params: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process depth map to create mapping for design warping."""
        x_min, x_max = design_params['x_min'], design_params['x_max']
        y_min, y_max = design_params['y_min'], design_params['y_max']
        dw, dh = design_params['original_dw'], design_params['original_dh']
        
        # Extract and normalize depth in the region
        D_box = depth_map[y_min:y_max, x_min:x_max].astype(float)
        
        # Check if depth map has valid values
        if D_box.min() == D_box.max():
            self.logger.warning(f"Depth map has uniform values: min={D_box.min()}, max={D_box.max()}")
            # Create a simple linear mapping as fallback
            map_x_box = np.tile(np.linspace(0, dw-1, D_box.shape[1]), (D_box.shape[0], 1))
            map_y_box = np.tile(np.linspace(0, dh-1, D_box.shape[0]).reshape(-1, 1), (1, D_box.shape[1]))
            return map_x_box, map_y_box
        
        D_box = (D_box - D_box.min()) / (D_box.max() - D_box.min() + 1e-6)  # Normalize to [0,1]
        
        # Save debug image of normalized depth
        save_debug_image(
            (D_box * 255).astype(np.uint8), 
            f"depth_normalized_{x_min}_{y_min}_{x_max}_{y_max}"
        )
        
        # Calculate gradients and scaling factors
        k = 1.0  # Bending strength
        Dy, Dx = np.gradient(D_box)
        
        # Save debug images of gradients
        save_debug_image(
            ((Dx + 1) * 127.5).astype(np.uint8), 
            f"depth_gradient_x_{x_min}_{y_min}_{x_max}_{y_max}"
        )
        save_debug_image(
            ((Dy + 1) * 127.5).astype(np.uint8), 
            f"depth_gradient_y_{x_min}_{y_min}_{x_max}_{y_max}"
        )
        
        sx = 1 / (1 + k * np.abs(Dx))
        sy = 1 / (1 + k * np.abs(Dy))
        
        # Compute cumulative sums
        map_x_box = np.cumsum(sx, axis=1)
        map_y_box = np.cumsum(sy, axis=0)
        
        # Check for division by zero
        total_sx = map_x_box[:, -1]
        total_sy = map_y_box[-1, :]
        
        # Replace zeros with small values to avoid division by zero
        total_sx[total_sx == 0] = 1e-6
        total_sy[total_sy == 0] = 1e-6
        
        # Scale to design dimensions
        scaled_map_x_box = (map_x_box / total_sx[:, np.newaxis]) * (dw - 1)
        scaled_map_y_box = (map_y_box / total_sy[np.newaxis, :]) * (dh - 1)
        
        return scaled_map_x_box, scaled_map_y_box
    
    @debug_timing
    @debug_exception
    def warp_design_with_depth(
        self, 
        source_depth: np.ndarray, 
        design_image: np.ndarray, 
        design_params: Dict[str, int]
    ) -> np.ndarray:
        """Warp the design using the depth map."""
        image_height, image_width = source_depth.shape[:2]
        x_min, x_max = design_params['x_min'], design_params['x_max']
        y_min, y_max = design_params['y_min'], design_params['y_max']
        
        try:
            # Process depth map to get mapping
            scaled_map_x_box, scaled_map_y_box = self.process_depth_map(source_depth, design_params)
            
            # Create full-size mapping arrays
            map_x = np.full((image_height, image_width), -1, dtype=np.float32)
            map_y = np.full((image_height, image_width), -1, dtype=np.float32)
            map_x[y_min:y_max, x_min:x_max] = scaled_map_x_box
            map_y[y_min:y_max, x_min:x_max] = scaled_map_y_box
            
            # Warp the 4-channel design (BGRA) with high-quality interpolation
            warped_design = cv2.remap(
                design_image, 
                map_x, 
                map_y, 
                cv2.INTER_LANCZOS4,  # Use Lanczos interpolation for higher quality
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(0, 0, 0, 0)
            )
            
            return warped_design
            
        except Exception as e:
            self.logger.error(f"Error in warping design: {str(e)}", exc_info=True)
            # Fallback: return original design placed at the target location
            result = np.zeros((image_height, image_width, 4), dtype=design_image.dtype)
            
            # Calculate placement coordinates
            design_h, design_w = design_image.shape[:2]
            scaled_h = design_params['scaled_dh']
            scaled_w = design_params['scaled_dw']
            
            # Resize design
            resized_design = cv2.resize(design_image, (scaled_w, scaled_h))
            
            # Place in the result
            result[y_min:y_max, x_min:x_max] = resized_design
            
            self.logger.warning("Using fallback design placement without warping")
            return result
    
    @debug_exception
    def apply_shading_to_design(
        self, 
        warped_design: np.ndarray, 
        source_image: np.ndarray, 
        source_mask: np.ndarray, 
        shading_strength: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply shading to the warped design based on the source image."""
        # Extract color and alpha channels
        warped_color = warped_design[:, :, :3]  # BGR channels
        warped_alpha = warped_design[:, :, 3].astype(float) / 255  # Alpha channel [0,1]
        
        # Calculate shading from source image
        shading = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).astype(float) / 255
        
        # Adjust design brightness based on shading and strength factor
        adjusted_shading = 1 - shading_strength + shading_strength * shading
        # Reshape to (height, width, 1) for proper broadcasting with 3-channel image
        adjusted_shading = adjusted_shading[:, :, np.newaxis]
        design_shaded_color = (warped_color.astype(float) * adjusted_shading).astype(np.uint8)
        
        # Combine alpha with mask
        effective_alpha = warped_alpha * (source_mask.astype(float) / 255)
        effective_alpha = effective_alpha[:, :, np.newaxis]  # Shape (h, w, 1)
        
        # Save debug images
        save_debug_image((adjusted_shading * 255).astype(np.uint8), "adjusted_shading")
        save_debug_image((effective_alpha * 255).astype(np.uint8), "effective_alpha")
        
        return design_shaded_color, effective_alpha 