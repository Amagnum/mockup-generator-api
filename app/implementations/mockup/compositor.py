import cv2
import numpy as np
from typing import Tuple
from app.utils.logging_config import get_logger
from app.utils.debug_utils import debug_exception
from app.utils.debug_utils import save_debug_image

class Compositor:
    """Handles image compositing for the mockup generator"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @debug_exception
    def composite_final_image(
        self, 
        design_shaded_color: np.ndarray, 
        colored_tshirt_image: np.ndarray, 
        effective_alpha: np.ndarray
    ) -> np.ndarray:
        """Composite the shaded design onto the colored t-shirt."""
        try:
            # Check dimensions
            if design_shaded_color.shape[:2] != colored_tshirt_image.shape[:2]:
                self.logger.error(f"Shape mismatch in composite: design={design_shaded_color.shape}, "
                                f"tshirt={colored_tshirt_image.shape}")
                return colored_tshirt_image  # Early return if dimensions don't match
            
            # Get the number of channels in the colored t-shirt image
            num_channels = colored_tshirt_image.shape[-1]
            
            # Handle channel mismatch between design and t-shirt
            if design_shaded_color.shape[-1] != num_channels:
                self.logger.debug(f"Adjusting design channels from {design_shaded_color.shape[-1]} to {num_channels}")
                
                # If design has 3 channels (BGR) and t-shirt has 4 (BGRA), add alpha channel
                if design_shaded_color.shape[-1] == 3 and num_channels == 4:
                    # Create a new array with 4 channels
                    design_with_alpha = np.zeros((*design_shaded_color.shape[:2], 4), dtype=design_shaded_color.dtype)
                    # Copy the BGR channels
                    design_with_alpha[:, :, :3] = design_shaded_color
                    # Set alpha channel to maximum (255)
                    design_with_alpha[:, :, 3] = 255
                    design_shaded_color = design_with_alpha
                    self.logger.debug(f"Added alpha channel to design, new shape: {design_shaded_color.shape}")
                elif design_shaded_color.shape[-1] > num_channels:
                    design_shaded_color = design_shaded_color[:, :, :num_channels]
                else:
                    # If design has fewer channels and not the BGR/BGRA case, log warning and return
                    self.logger.warning(f"Design has fewer channels ({design_shaded_color.shape[-1]}) than t-shirt ({num_channels}) and not in expected format")
                    return colored_tshirt_image
            
            # Process effective_alpha to match the number of channels
            # First, ensure it's a 3D array with shape (h, w, 1)
            if effective_alpha.ndim == 2:
                effective_alpha = effective_alpha[:, :, np.newaxis]
            
            # Now expand to match the number of channels if needed
            if effective_alpha.shape[-1] == 1:
                effective_alpha_expanded = np.repeat(effective_alpha, num_channels, axis=2)
            elif effective_alpha.shape[-1] == num_channels:
                effective_alpha_expanded = effective_alpha
            elif effective_alpha.shape[-1] > num_channels:
                effective_alpha_expanded = effective_alpha[:, :, :num_channels]
            else:
                # Unusual case - alpha has more than 1 but fewer than num_channels
                self.logger.warning(f"Alpha has unusual number of channels: {effective_alpha.shape[-1]}")
                # Repeat the first channel to match num_channels
                effective_alpha_expanded = np.repeat(effective_alpha[:, :, :1], num_channels, axis=2)
            
            # Check for NaN or Inf values
            if np.isnan(effective_alpha_expanded).any() or np.isinf(effective_alpha_expanded).any():
                self.logger.warning("Alpha channel contains NaN or Inf values, replacing with zeros")
                effective_alpha_expanded = np.nan_to_num(effective_alpha_expanded)
            
            # Clip alpha values to ensure they're in the valid range [0, 1]
            effective_alpha_expanded = np.clip(effective_alpha_expanded, 0.0, 1.0)
            
            # Log shapes and alpha statistics for debugging
            self.logger.debug(f"Compositing shapes - design: {design_shaded_color.shape}, " 
                             f"tshirt: {colored_tshirt_image.shape}, "
                             f"alpha: {effective_alpha_expanded.shape}")
            self.logger.debug(f"Alpha stats - min: {effective_alpha_expanded.min()}, " 
                             f"max: {effective_alpha_expanded.max()}, "
                             f"mean: {effective_alpha_expanded.mean()}")
            
            # Save debug image of the alpha channel
            save_debug_image((effective_alpha_expanded[:,:,0] * 255).astype(np.uint8), "final_alpha_channel")
            
            # Final check before blending
            if (design_shaded_color.shape != colored_tshirt_image.shape or 
                effective_alpha_expanded.shape != colored_tshirt_image.shape):
                self.logger.error(f"Shape mismatch before blending: "
                                f"design={design_shaded_color.shape}, "
                                f"tshirt={colored_tshirt_image.shape}, "
                                f"alpha={effective_alpha_expanded.shape}")
                return colored_tshirt_image  # Early return if shapes don't match
            
            # Convert to float for blending to avoid overflow
            design_float = design_shaded_color.astype(float)
            tshirt_float = colored_tshirt_image.astype(float)
            
            # Perform alpha blending
            final_image = (
                effective_alpha_expanded * design_float +
                (1.0 - effective_alpha_expanded) * tshirt_float
            ).astype(np.uint8)
            
            # Save intermediate results for debugging
            save_debug_image(design_shaded_color, "design_before_composite")
            save_debug_image(colored_tshirt_image, "tshirt_before_composite")
            save_debug_image(final_image, "final_after_composite")
            
            return final_image
            
        except Exception as e:
            self.logger.error(f"Error in compositing: {str(e)}", exc_info=True)
            # Return colored t-shirt as fallback
            return colored_tshirt_image 