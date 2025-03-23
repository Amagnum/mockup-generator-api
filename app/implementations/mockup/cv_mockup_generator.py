import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional

from app.interfaces.mockup_generator import MockupGenerator
from app.utils.logging_config import get_logger
from app.utils.debug_utils import save_debug_image, debug_timing, debug_exception, debug_function_args

from app.implementations.mockup.color_handler import ColorHandler
from app.implementations.mockup.design_placer import DesignPlacer
from app.implementations.mockup.design_warper import DesignWarper
from app.implementations.mockup.compositor import Compositor

class CVMockupGenerator(MockupGenerator):
    """OpenCV-based implementation of the mockup generator"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.color_handler = ColorHandler()
        self.design_placer = DesignPlacer()
        self.design_warper = DesignWarper()
        self.compositor = Compositor()
    
    @debug_timing
    @debug_exception
    def generate_tshirt_mockup(
        self,
        source_image: np.ndarray,
        source_mask: np.ndarray,
        source_depth: np.ndarray,
        design_image: np.ndarray,
        color_code: Tuple[int, int, int],
        location: Tuple[int, int],
        scale_factor: float,
        shading_strength: float = 1.0,
        color_mode: str = 'auto',
        color_method: str = 'standard',
        color_config: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Generate a t-shirt mockup with a design placed at a 2D location, bent using a 16-bit depth map.

        Parameters:
        - source_image: BGR image of a person in a white t-shirt.
        - source_mask: Binary mask (255 for t-shirt, 0 elsewhere).
        - source_depth: 16-bit depth map (grayscale, 0–65535).
        - design_image: BGRA image of the design (with alpha channel).
        - color_code: RGB tuple for t-shirt color (e.g., (255, 0, 0)).
        - location: Tuple (x, y) for design center.
        - scale_factor: Float to scale the design size.
        - shading_strength: Float (0 to 1) to control shading intensity on the design.
        - color_mode: String, one of 'auto', 'light', or 'dark' to control color application method.
        - color_method: String, one of 'standard' or 'intrinsic' to select coloring algorithm.
        - color_config: Dictionary with parameters to control the recoloring process.

        Returns:
        - final_image: BGR image with colored t-shirt and design applied.
        """
        # Create a unique debug ID for this mockup generation
        debug_id = f"debug_{id(source_image)}"
        self.logger.info(f"Starting mockup generation with debug_id: {debug_id}")
        
        # Validate inputs
        self._validate_inputs(source_image, source_mask, source_depth, design_image)
        
        # Step 1: Change T-shirt Color
        self.logger.debug(f"Step 1: Applying t-shirt color using method: {color_method}")
        colored_tshirt_image = self.color_handler.apply_tshirt_color(
            source_image, 
            source_mask, 
            color_code, 
            color_mode,
            method=color_method,
            config=color_config
        )
        save_debug_image(colored_tshirt_image, f"{debug_id}_1_colored_tshirt")
        
        # Step 2: Compute design placement parameters
        self.logger.debug("Step 2: Computing design placement")
        design_params = self.design_placer.compute_design_placement(
            source_image.shape[:2], 
            design_image.shape[:2], 
            location, 
            scale_factor
        )
        self.logger.debug(f"Design placement parameters: {design_params}")
        
        # Step 3-4: Process depth map and warp design
        self.logger.debug("Step 3-4: Warping design with depth map")
        warped_design = self.design_warper.warp_design_with_depth(
            source_depth, 
            design_image, 
            design_params
        )
        save_debug_image(warped_design, f"{debug_id}_2_warped_design")
        
        # Step 5: Apply shading to design
        self.logger.debug("Step 5: Applying shading to design")
        design_shaded_color, effective_alpha = self.design_warper.apply_shading_to_design(
            warped_design, 
            source_image, 
            source_mask, 
            shading_strength
        )
        save_debug_image(design_shaded_color, f"{debug_id}_3_shaded_design")
        
        # Step 6: Composite final image
        self.logger.debug("Step 6: Compositing final image")
        final_image = self.compositor.composite_final_image(
            design_shaded_color, 
            colored_tshirt_image, 
            effective_alpha
        )
        save_debug_image(final_image, f"{debug_id}_4_final_image")
        
        self.logger.info(f"Mockup generation completed successfully")
        return final_image
    
    @debug_exception
    def _validate_inputs(
        self, 
        source_image: Optional[np.ndarray], 
        source_mask: Optional[np.ndarray], 
        source_depth: Optional[np.ndarray], 
        design_image: Optional[np.ndarray]
    ) -> None:
        """Validate that all required inputs are provided and valid."""
        if source_image is None or source_mask is None or source_depth is None or design_image is None:
            self.logger.error("One or more input images are None")
            raise ValueError("One or more input images are None")
            
        # Log image properties
        self.logger.debug(f"Source image: shape={source_image.shape}, dtype={source_image.dtype}")
        self.logger.debug(f"Source mask: shape={source_mask.shape}, dtype={source_mask.dtype}")
        self.logger.debug(f"Source depth: shape={source_depth.shape}, dtype={source_depth.dtype}")
        self.logger.debug(f"Design image: shape={design_image.shape}, dtype={design_image.dtype}")
        
        # Check if mask is binary
        if not np.array_equal(np.unique(source_mask), np.array([0, 255])) and not np.array_equal(np.unique(source_mask), np.array([0])) and not np.array_equal(np.unique(source_mask), np.array([255])):
            unique_values = np.unique(source_mask)
            self.logger.warning(f"Mask is not binary. Unique values: {unique_values}") 