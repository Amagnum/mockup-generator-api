import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional

from app.interfaces.mockup_generator import MockupGenerator

class CVMockupGenerator(MockupGenerator):
    """OpenCV-based implementation of the mockup generator"""
    
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
        # Validate inputs
        self._validate_inputs(source_image, source_mask, source_depth, design_image)
        
        # Step 1: Change T-shirt Color
        colored_tshirt_image = self._apply_tshirt_color(
            source_image, 
            source_mask, 
            color_code, 
            color_mode
        )
        
        # Step 2: Compute design placement parameters
        design_params = self._compute_design_placement(
            source_image.shape[:2], 
            design_image.shape[:2], 
            location, 
            scale_factor
        )
        
        # Step 3-4: Process depth map and warp design
        warped_design = self._warp_design_with_depth(
            source_depth, 
            design_image, 
            design_params
        )
        
        # Step 5: Apply shading to design
        design_shaded_color, effective_alpha = self._apply_shading_to_design(
            warped_design, 
            source_image, 
            source_mask, 
            shading_strength
        )
        
        # Step 6: Composite final image
        final_image = self._composite_final_image(
            design_shaded_color, 
            colored_tshirt_image, 
            effective_alpha
        )
        
        return final_image
    
    def _validate_inputs(
        self, 
        source_image: Optional[np.ndarray], 
        source_mask: Optional[np.ndarray], 
        source_depth: Optional[np.ndarray], 
        design_image: Optional[np.ndarray]
    ) -> None:
        """Validate that all required inputs are provided and valid."""
        if source_image is None or source_mask is None or source_depth is None or design_image is None:
            raise ValueError("One or more input images are None")
    
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
        return 'light' if brightness > 128 else 'dark'
    
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
    
    def _apply_light_color(
        self, 
        source_image: np.ndarray, 
        mask: np.ndarray, 
        color_code: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply light color using HSV color space to preserve shading."""
        # Convert RGB color_code to HSV
        color_rgb = np.array([[color_code]], dtype=np.uint8)
        color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        
        # Convert source image to HSV and apply color
        hsv_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
        hsv_source[mask, 0] = color_hsv[0]  # Hue
        hsv_source[mask, 1] = color_hsv[1]  # Saturation
        
        return cv2.cvtColor(hsv_source, cv2.COLOR_HSV2BGR)
    
    def _apply_dark_color(
        self, 
        source_image: np.ndarray, 
        mask: np.ndarray, 
        color_code: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply dark color using multiply blending for better results."""
        # Create a copy of the source image
        colored_image = source_image.copy()
        
        # Create a solid color image with the target color
        solid_color = np.zeros_like(source_image)
        solid_color[:] = (color_code[2], color_code[1], color_code[0])  # BGR format
        
        # Extract lighting information from the original t-shirt
        luminance = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        
        # Apply color with lighting preservation
        for c in range(3):  # BGR channels
            # Multiply mode blending: preserves shadows and highlights
            colored_image[mask, c] = np.clip(
                (solid_color[mask, c].astype(float) * luminance[mask] * 1.5),
                0,
                255
            ).astype(np.uint8)
            
        return colored_image
    
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
        
        if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            raise ValueError("Design placement exceeds image boundaries.")
            
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
        D_box = (D_box - D_box.min()) / (D_box.max() - D_box.min() + 1e-6)  # Normalize to [0,1]
        
        # Calculate gradients and scaling factors
        k = 1.0  # Bending strength
        Dy, Dx = np.gradient(D_box)
        sx = 1 / (1 + k * np.abs(Dx))
        sy = 1 / (1 + k * np.abs(Dy))
        
        # Compute cumulative sums
        map_x_box = np.cumsum(sx, axis=1)
        map_y_box = np.cumsum(sy, axis=0)
        
        # Scale to design dimensions
        total_sx = map_x_box[:, -1]
        total_sy = map_y_box[-1, :]
        total_sx[total_sx == 0] = 1e-6
        total_sy[total_sy == 0] = 1e-6
        
        scaled_map_x_box = (map_x_box / total_sx[:, np.newaxis]) * (dw - 1)
        scaled_map_y_box = (map_y_box / total_sy[np.newaxis, :]) * (dh - 1)
        
        return scaled_map_x_box, scaled_map_y_box
    
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
        
        # Process depth map to get mapping
        scaled_map_x_box, scaled_map_y_box = self._process_depth_map(source_depth, design_params)
        
        # Create full-size mapping arrays
        map_x = np.full((image_height, image_width), -1, dtype=np.float32)
        map_y = np.full((image_height, image_width), -1, dtype=np.float32)
        map_x[y_min:y_max, x_min:x_max] = scaled_map_x_box
        map_y[y_min:y_max, x_min:x_max] = scaled_map_y_box
        
        # Warp the 4-channel design (BGRA)
        warped_design = cv2.remap(
            design_image, 
            map_x, 
            map_y, 
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0, 0)
        )
        
        return warped_design
    
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
        design_shaded_color = (warped_color.astype(float) * adjusted_shading[:, :, np.newaxis]).astype(np.uint8)
        
        # Combine alpha with mask
        effective_alpha = warped_alpha * (source_mask.astype(float) / 255)
        effective_alpha = effective_alpha[:, :, np.newaxis]  # Shape (h, w, 1)
        
        return design_shaded_color, effective_alpha
    
    def _composite_final_image(
        self, 
        design_shaded_color: np.ndarray, 
        colored_tshirt_image: np.ndarray, 
        effective_alpha: np.ndarray
    ) -> np.ndarray:
        """Composite the shaded design onto the colored t-shirt."""
        final_image = (
            effective_alpha * design_shaded_color +
            (1 - effective_alpha) * colored_tshirt_image
        ).astype(np.uint8)
        
        return final_image 