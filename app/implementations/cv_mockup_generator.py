import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional

from app.interfaces.mockup_generator import MockupGenerator
from app.utils.logging_config import get_logger
from app.utils.debug_utils import save_debug_image, debug_timing, debug_exception, debug_function_args

class CVMockupGenerator(MockupGenerator):
    """OpenCV-based implementation of the mockup generator"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
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
        color_mode: str = 'auto'
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

        Returns:
        - final_image: BGR image with colored t-shirt and design applied.
        """
        # Create a unique debug ID for this mockup generation
        debug_id = f"debug_{id(source_image)}"
        self.logger.info(f"Starting mockup generation with debug_id: {debug_id}")
        
        # Validate inputs
        self._validate_inputs(source_image, source_mask, source_depth, design_image)
        
        # Step 1: Change T-shirt Color
        self.logger.debug("Step 1: Applying t-shirt color")
        colored_tshirt_image = self._apply_tshirt_color(
            source_image, 
            source_mask, 
            color_code, 
            color_mode
        )
        save_debug_image(colored_tshirt_image, f"{debug_id}_1_colored_tshirt")
        
        # Step 2: Compute design placement parameters
        self.logger.debug("Step 2: Computing design placement")
        design_params = self._compute_design_placement(
            source_image.shape[:2], 
            design_image.shape[:2], 
            location, 
            scale_factor
        )
        self.logger.debug(f"Design placement parameters: {design_params}")
        
        # Step 3-4: Process depth map and warp design
        self.logger.debug("Step 3-4: Warping design with depth map")
        warped_design = self._warp_design_with_depth(
            source_depth, 
            design_image, 
            design_params
        )
        save_debug_image(warped_design, f"{debug_id}_2_warped_design")
        
        # Step 5: Apply shading to design
        self.logger.debug("Step 5: Applying shading to design")
        design_shaded_color, effective_alpha = self._apply_shading_to_design(
            warped_design, 
            source_image, 
            source_mask, 
            shading_strength
        )
        save_debug_image(design_shaded_color, f"{debug_id}_3_shaded_design")
        
        # Step 6: Composite final image
        self.logger.debug("Step 6: Compositing final image")
        final_image = self._composite_final_image(
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
    
    @debug_function_args
    def _determine_color_mode(
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
    def _apply_tshirt_color(
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
        color_mode = self._determine_color_mode(color_code, color_mode)
        
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
    
    @debug_exception
    def _compute_design_placement(
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
    
    @debug_timing
    @debug_exception
    def _process_depth_map(
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
    def _warp_design_with_depth(
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
            scaled_map_x_box, scaled_map_y_box = self._process_depth_map(source_depth, design_params)
            
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
    def _apply_shading_to_design(
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
    
    @debug_exception
    def _composite_final_image(
        self,
        design_shaded_color: np.ndarray,
        colored_tshirt_image: np.ndarray,
        effective_alpha: np.ndarray
    ) -> np.ndarray:
        """
        Composite the shaded design onto the colored t-shirt.
        
        This function handles edge cases including:
        - Mismatched image dimensions (height/width)
        - Different number of channels between design and t-shirt images
        - Effective alpha channel shape adjustments and invalid values (NaN/Inf)
        - Proper type conversions and clipping for safe blending.
        """
        # Check height and width dimensions
        if design_shaded_color.shape[:2] != colored_tshirt_image.shape[:2]:
            self.logger.error(
                "Shape mismatch in composite: design=%s, tshirt=%s",
                design_shaded_color.shape,
                colored_tshirt_image.shape
            )
            # Fallback: return the t-shirt image if dimensions differ.
            return colored_tshirt_image

        # Determine target channel count from the t-shirt image
        target_channels = colored_tshirt_image.shape[-1]

        # Adjust design_shaded_color channels to match colored_tshirt_image channels
        if design_shaded_color.shape[-1] != target_channels:
            self.logger.debug(
                "Adjusting design channels from %d to %d",
                design_shaded_color.shape[-1],
                target_channels
            )
            if design_shaded_color.shape[-1] > target_channels:
                # Slice extra channels off
                design_shaded_color = design_shaded_color[:, :, :target_channels]
            else:
                # If there are fewer channels, pad by replicating the last channel
                missing = target_channels - design_shaded_color.shape[-1]
                pad = np.repeat(design_shaded_color[:, :, -1:], missing, axis=2)
                design_shaded_color = np.concatenate([design_shaded_color, pad], axis=2)
            self.logger.debug("New design_shaded_color shape: %s", design_shaded_color.shape)

        # Validate effective_alpha dimensions
        if effective_alpha.shape[:2] != colored_tshirt_image.shape[:2]:
            self.logger.error(
                "Effective alpha shape mismatch: effective_alpha=%s, expected=%s",
                effective_alpha.shape,
                colored_tshirt_image.shape
            )
            return colored_tshirt_image

        # If effective_alpha is 2D (height x width), add a channel dimension
        if effective_alpha.ndim == 2:
            effective_alpha = effective_alpha[:, :, np.newaxis]

        # Expand or adjust effective_alpha channels to match target_channels
        if effective_alpha.shape[-1] == 1:
            effective_alpha_expanded = np.repeat(effective_alpha, target_channels, axis=2)
        elif effective_alpha.shape[-1] != target_channels:
            self.logger.debug(
                "Adjusting effective_alpha channels from %d to %d",
                effective_alpha.shape[-1],
                target_channels
            )
            effective_alpha_expanded = effective_alpha[:, :, :target_channels]
        else:
            effective_alpha_expanded = effective_alpha

        # Check and fix any NaN or Inf values in the alpha channel
        if np.isnan(effective_alpha_expanded).any() or np.isinf(effective_alpha_expanded).any():
            self.logger.warning("Alpha channel contains NaN or Inf values, replacing with zeros")
            effective_alpha_expanded = np.nan_to_num(effective_alpha_expanded)

        # Clip effective_alpha values to the range [0, 1]
        effective_alpha_expanded = np.clip(effective_alpha_expanded, 0, 1)

        # Log shapes for debugging
        self.logger.debug(
            "Compositing shapes - design: %s, tshirt: %s, alpha: %s",
            design_shaded_color.shape,
            colored_tshirt_image.shape,
            effective_alpha_expanded.shape
        )

        # Convert images to float32 for safe arithmetic
        design_float = design_shaded_color.astype(np.float32)
        tshirt_float = colored_tshirt_image.astype(np.float32)

        try:
            # Perform the alpha blending
            final_float = (
                effective_alpha_expanded * design_float +
                (1 - effective_alpha_expanded) * tshirt_float
            )
            # Ensure pixel values remain within valid range and convert to uint8
            final_float = np.clip(final_float, 0, 255)
            final_image = final_float.astype(np.uint8)
            return final_image
        except Exception as e:
            self.logger.error("Error in compositing: %s", str(e), exc_info=True)
            # Return the colored t-shirt image as a fallback
            return colored_tshirt_image
