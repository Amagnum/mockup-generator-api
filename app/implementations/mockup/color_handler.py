import cv2
import numpy as np
from typing import Tuple, Dict, Any, Literal, Union
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
        color_mode: str,
        method: str = 'standard',
        config: Dict[str, Any] = None
    ) -> np.ndarray:
        """Apply color to the t-shirt area using appropriate blending method.
        
        Args:
            source_image: Original image
            source_mask: Mask of the t-shirt area
            color_code: RGB color tuple (R, G, B)
            color_mode: 'light', 'dark', or 'auto'
            method: Coloring method to use ('standard', 'intrinsic', 'lum-v2', or 'int-v2')
            config: Optional dictionary with parameters to control the recoloring process
        
        Returns:
            Colored t-shirt image
        """
        # Create a copy of the source image
        colored_tshirt_image = source_image.copy()
        
        # Extract mask
        mask = source_mask > 0
        
        if method == 'intrinsic':
            self.logger.debug("Using intrinsic image decomposition method")
            return self._apply_intrinsic_coloring(source_image, mask, color_code)
        elif method == 'lum-v2':
            self.logger.debug("Using luminance-based v2 coloring method")
            return self._apply_lum_v2_coloring(source_image, mask, color_code)
        elif method == 'int-v2':
            self.logger.debug("Using enhanced intrinsic decomposition method (int-v2)")
            return self._apply_int_v2_coloring(source_image, mask, color_code, config)
        else:
            # Standard method - determine color application method based on light/dark
            color_mode = self.determine_color_mode(color_code, color_mode)
            
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
    def _apply_intrinsic_coloring(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply color using intrinsic image decomposition to preserve texture details.
        
        This method separates the image into albedo (base color) and shading components,
        then applies the new color while preserving the original shading details.
        
        Args:
            source_image: Original image (BGR format)
            mask: Boolean mask of the t-shirt area
            color_code: RGB color tuple (R, G, B)
            
        Returns:
            Recolored image with preserved texture details
        """
        self.logger.debug("Applying color using intrinsic image decomposition")
        
        # Create a copy of the source image
        colored_image = source_image.copy()
        
        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        
        # Convert to float for calculations
        rgb_float = rgb_image.astype(np.float32) / 255.0
        
        # Calculate luminance using perceptual weights
        luminance = np.dot(rgb_float[..., :3], [0.299, 0.587, 0.114])
        
        # Enhanced shading estimation with boosted contrast
        shading = np.clip(luminance * 1.2, 0, 1)
        
        # Convert target RGB color to BGR for OpenCV
        target_bgr = (color_code[2], color_code[1], color_code[0])
        
        # Apply color with texture preservation
        for c in range(3):  # BGR channels
            # Apply the 85% target color + 15% original luminance + 10% shading boost formula
            colored_image[mask, c] = np.clip(
                ((target_bgr[c] / 255.0 * 0.90) + (luminance[mask] * 0.10)) * (shading[mask] * 1.1) * 255.0,
                0,
                255
            ).astype(np.uint8)
        
        return colored_image
    
    @debug_exception
    def _apply_lum_v2_coloring(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply color using luminance-based recoloring (v2) to preserve texture details.
        
        This method extracts the luminance from the original image and multiplies it with
        the target color to preserve shading and texture details.
        
        Args:
            source_image: Original image (BGR format)
            mask: Boolean mask of the t-shirt area
            color_code: RGB color tuple (R, G, B)
            
        Returns:
            Recolored image with preserved texture details
        """
        self.logger.debug("Applying color using luminance-based v2 method")
        
        # Create a copy of the source image
        colored_image = source_image.copy()
        
        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        
        # Convert to float for calculations
        rgb_float = rgb_image.astype(np.float32) / 255.0
        
        # Normalize the target color to [0, 1] range
        target_color_norm = np.array(color_code, dtype=np.float32) / 255.0
        
        # Extract the luminance by converting to grayscale
        luminance = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2GRAY)
        
        # Create a 3D mask for easier processing
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # Apply the recoloring only to the masked area
        # Convert RGB to BGR for OpenCV
        for c in range(3):
            # For BGR format, we need to reverse the channel index
            target_channel = target_color_norm[2-c]
            colored_image[mask, c] = np.clip(
                luminance[mask] * target_channel * 255.0,
                0,
                255
            ).astype(np.uint8)
        
        return colored_image
    
    @debug_exception
    def _apply_int_v2_coloring(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int],
        config: Dict[str, Any] = None
    ) -> np.ndarray:
        """Apply color using enhanced intrinsic image decomposition (int-v2).
        
        This method improves upon the original intrinsic decomposition by:
        1. Using a multi-scale approach to better separate texture from shading
        2. Applying adaptive color blending based on local texture complexity
        3. Preserving fine details through frequency-based decomposition
        
        Args:
            source_image: Original image (BGR format)
            mask: Boolean mask of the t-shirt area
            color_code: RGB color tuple (R, G, B)
            config: Optional dictionary with parameters to control the recoloring:
                - large_scale_blur: Size of kernel for large-scale shading (default: 21)
                - medium_scale_blur: Size of kernel for medium details (default: 7)
                - fine_scale_blur: Size of kernel for fine details (default: 3)
                - large_scale_weight: Weight for large-scale shading (default: 0.7)
                - medium_scale_weight: Weight for medium details (default: 1.5)
                - fine_scale_weight: Weight for fine details (default: 2.0)
                - min_shading: Minimum shading value to prevent black areas (default: 0.05)
                - shading_boost: Boost factor for final shading (default: 1.2)
                - base_detail_preservation: Base level of detail to preserve (default: 0.15)
                - texture_detail_weight: How much additional detail to preserve in textured areas (default: 0.25)
                - saturation_influence: How much color saturation reduces detail preservation (default: 0.3)
            
        Returns:
            Recolored image with enhanced texture preservation
        """
        self.logger.debug("Applying color using enhanced intrinsic decomposition (int-v2)")
        
        # Set default configuration parameters
        default_config = {
            'large_scale_blur': 21,
            'medium_scale_blur': 7,
            'fine_scale_blur': 3,
            'large_scale_weight': 0.7,
            'medium_scale_weight': 1.5,
            'fine_scale_weight': 2.0,
            'min_shading': 0.05,
            'shading_boost': 1.2,
            'base_detail_preservation': 0.15,
            'texture_detail_weight': 0.25,
            'saturation_influence': 0.3
        }
        
        # Use provided config or defaults
        if config is None:
            config = {}
        
        # Merge with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        # Create a copy of the source image
        colored_image = source_image.copy()
        
        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        
        # Convert to float for calculations
        rgb_float = rgb_image.astype(np.float32) / 255.0
        
        # Calculate base luminance using perceptual weights
        base_luminance = np.dot(rgb_float[..., :3], [0.299, 0.587, 0.114])
        
        # Ensure base_luminance is properly formatted for OpenCV operations
        base_luminance_img = base_luminance.astype(np.float32)
        
        # Multi-scale decomposition for better texture/shading separation
        # Create a blurred version for large-scale shading
        large_kernel = config['large_scale_blur']
        if large_kernel % 2 == 0:  # Ensure kernel size is odd
            large_kernel += 1
        blurred = cv2.GaussianBlur(base_luminance_img, (large_kernel, large_kernel), 0)
        
        # Extract medium details
        medium_kernel = config['medium_scale_blur']
        if medium_kernel % 2 == 0:
            medium_kernel += 1
        medium_details = cv2.GaussianBlur(base_luminance_img, (medium_kernel, medium_kernel), 0) - blurred
        
        # Extract fine details (high frequency)
        fine_kernel = config['fine_scale_blur']
        if fine_kernel % 2 == 0:
            fine_kernel += 1
        fine_details = base_luminance_img - cv2.GaussianBlur(base_luminance_img, (fine_kernel, fine_kernel), 0)
        
        # Combine into enhanced shading map with detail preservation
        shading = np.clip(
            blurred * config['large_scale_weight'] + 
            medium_details * config['medium_scale_weight'] + 
            fine_details * config['fine_scale_weight'], 
            config['min_shading'], 
            1.0
        )
        
        # Calculate local texture complexity for adaptive blending
        texture_complexity = np.abs(cv2.Laplacian(base_luminance_img, cv2.CV_32F, ksize=3))
        texture_complexity = cv2.GaussianBlur(texture_complexity, (5, 5), 0)
        
        # Normalize texture complexity
        if texture_complexity.max() > 0:  # Avoid division by zero
            texture_complexity = np.clip(texture_complexity / texture_complexity.max(), 0, 1)
        else:
            texture_complexity = np.zeros_like(texture_complexity)
        
        # Convert target RGB color to BGR for OpenCV
        target_bgr = (color_code[2], color_code[1], color_code[0])
        
        # Calculate color saturation for adaptive blending
        color_hsv = cv2.cvtColor(np.uint8([[color_code]]), cv2.COLOR_RGB2HSV)[0][0]
        color_saturation = color_hsv[1] / 255.0  # Normalize to [0,1]
        
        # Apply color with enhanced texture preservation
        for c in range(3):  # BGR channels
            # Calculate adaptive blend factor (higher for complex textures)
            detail_preservation = config['base_detail_preservation'] + (
                config['texture_detail_weight'] * texture_complexity[mask]
            )
            
            # Adjust based on color saturation (more saturated colors need less detail preservation)
            detail_preservation *= (1.0 - (color_saturation * config['saturation_influence']))
            
            # Apply the enhanced formula with adaptive blending
            colored_image[mask, c] = np.clip(
                ((target_bgr[c] / 255.0 * (1.0 - detail_preservation)) + 
                 (rgb_float[mask, 2-c] * detail_preservation)) * 
                (shading[mask] * config['shading_boost']) * 255.0,
                0,
                255
            ).astype(np.uint8)
        
        return colored_image 