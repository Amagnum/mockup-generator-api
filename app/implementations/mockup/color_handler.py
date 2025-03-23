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
    
    @debug_function_args
    def detect_tshirt_base_color(
        self,
        source_image: np.ndarray,
        source_mask: np.ndarray
    ) -> str:
        """Detect if the source t-shirt is black or white.
        
        Args:
            source_image: Original image
            source_mask: Mask of the t-shirt area
            
        Returns:
            'white' or 'black' based on the detected base color
        """
        # Extract only the t-shirt area using the mask
        mask = source_mask > 0
        if not np.any(mask):
            self.logger.warning("Empty mask provided for t-shirt base color detection")
            return 'white'  # Default to white if mask is empty
            
        # Extract the t-shirt pixels
        tshirt_pixels = source_image[mask]
        
        # Check if the image has an alpha channel (4 channels)
        if tshirt_pixels.shape[1] == 4:
            # Use only the RGB channels for brightness calculation
            tshirt_pixels_rgb = tshirt_pixels[:, :3]
        else:
            tshirt_pixels_rgb = tshirt_pixels
        
        # Convert to RGB if the image is in BGR format (OpenCV default)
        if source_image.shape[2] >= 3:  # Could be 3 (RGB/BGR) or 4 (RGBA/BGRA)
            # For brightness calculation, we need RGB order
            # If it's BGR or BGRA, we need to swap R and B channels
            tshirt_pixels_rgb = tshirt_pixels_rgb[:, [2, 1, 0]] if tshirt_pixels_rgb.shape[1] == 3 else tshirt_pixels_rgb
        
        # Calculate the average brightness of the t-shirt using RGB channels
        avg_brightness = np.mean(np.dot(tshirt_pixels_rgb.astype(np.float32) / 255.0, [0.299, 0.587, 0.114]))
        
        # Calculate the median brightness (more robust to outliers)
        median_brightness = np.median(np.dot(tshirt_pixels_rgb.astype(np.float32) / 255.0, [0.299, 0.587, 0.114]))
        
        # Calculate standard deviation to detect if the t-shirt has a lot of variation
        std_brightness = np.std(np.dot(tshirt_pixels_rgb.astype(np.float32) / 255.0, [0.299, 0.587, 0.114]))
        
        self.logger.debug(f"T-shirt brightness - avg: {avg_brightness:.3f}, median: {median_brightness:.3f}, std: {std_brightness:.3f}")
        
        # Use median brightness for more robust detection
        # If median brightness is low, it's likely a dark/black t-shirt
        # The threshold is set at 0.3 (on a 0-1 scale) to account for shadows on white shirts
        base_color = 'black' if median_brightness < 0.3 else 'white'
        
        self.logger.info(f"Detected t-shirt base color: {base_color}")
        return base_color
    
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
        
        # Detect if the source t-shirt is black or white
        base_color = self.detect_tshirt_base_color(source_image, source_mask)
        self.logger.debug(f"Using {base_color} t-shirt as base for coloring")
        
        if method == 'int-v2':
            self.logger.debug("Using enhanced intrinsic decomposition method (int-v2)")
            if base_color == 'black':
                return self._apply_int_v2_coloring_black_base(source_image, mask, color_code, config)
            else:
                return self._apply_int_v2_coloring(source_image, mask, color_code, config)
        else:
            self.logger.debug("Using enhanced intrinsic decomposition method (int-v2)")
            if base_color == 'black':
                return self._apply_int_v2_coloring_black_base(source_image, mask, color_code, config)
            else:
                return self._apply_int_v2_coloring(source_image, mask, color_code, config)
    
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
    
    @debug_exception
    def _apply_int_v2_coloring_black_base(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int],
        config: Dict[str, Any] = None
    ) -> np.ndarray:
        """Apply color using enhanced intrinsic image decomposition for black t-shirts.
        
        This method is specifically designed for black or dark t-shirts, using an inverted
        approach to extract highlights and texture details from dark fabrics.
        
        Args:
            source_image: Original image (BGR format)
            mask: Boolean mask of the t-shirt area
            color_code: RGB color tuple (R, G, B)
            config: Optional dictionary with parameters to control the recoloring
            
        Returns:
            Recolored image with enhanced texture preservation for dark base fabrics
        """
        self.logger.debug("Applying color using enhanced intrinsic decomposition for black t-shirts")
        
        # Set default configuration parameters for black t-shirts
        default_config = {
            'large_scale_blur': 21,
            'medium_scale_blur': 7,
            'fine_scale_blur': 3,
            'large_scale_weight': 0.7,
            'medium_scale_weight': 1.5,
            'fine_scale_weight': 2.0,
            'highlight_boost': 1.5,        # Boost highlights in dark fabrics
            'shadow_compression': 0.7,     # Compress shadows to prevent too dark areas
            'base_detail_preservation': 0.2,
            'texture_detail_weight': 0.3,
            'saturation_influence': 0.25,
            'highlight_threshold': 0.4,    # Threshold to identify highlights in dark fabric
            'inversion_strength': 0.8,     # How strongly to invert the luminance for processing
            'color_intensity': 1.2         # Intensity of the target color (higher = more vibrant)
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
        
        # For black t-shirts, we need to invert the luminance to better extract details
        # This helps identify highlights and texture in dark fabrics
        inverted_luminance = 1.0 - base_luminance
        
        # Blend original and inverted luminance based on inversion strength
        # This preserves some of the original shading while enhancing details
        processed_luminance = (inverted_luminance * config['inversion_strength'] + 
                              base_luminance * (1.0 - config['inversion_strength']))
        
        # Ensure luminance is properly formatted for OpenCV operations
        luminance_img = processed_luminance.astype(np.float32)
        
        # Multi-scale decomposition for better texture/shading separation
        # Create a blurred version for large-scale shading
        large_kernel = config['large_scale_blur']
        if large_kernel % 2 == 0:  # Ensure kernel size is odd
            large_kernel += 1
        blurred = cv2.GaussianBlur(luminance_img, (large_kernel, large_kernel), 0)
        
        # Extract medium details
        medium_kernel = config['medium_scale_blur']
        if medium_kernel % 2 == 0:
            medium_kernel += 1
        medium_details = cv2.GaussianBlur(luminance_img, (medium_kernel, medium_kernel), 0) - blurred
        
        # Extract fine details (high frequency)
        fine_kernel = config['fine_scale_blur']
        if fine_kernel % 2 == 0:
            fine_kernel += 1
        fine_details = luminance_img - cv2.GaussianBlur(luminance_img, (fine_kernel, fine_kernel), 0)
        
        # Create a highlight mask to identify areas that should be brighter
        highlight_mask = base_luminance > config['highlight_threshold']
        
        # Combine into enhanced shading map with detail preservation
        shading = blurred * config['large_scale_weight'] + \
                 medium_details * config['medium_scale_weight'] + \
                 fine_details * config['fine_scale_weight']
        
        # Boost highlights in the fabric
        if np.any(highlight_mask):
            shading[highlight_mask] *= config['highlight_boost']
        
        # Compress shadows to prevent too dark areas
        shading = 1.0 - ((1.0 - shading) * config['shadow_compression'])
        
        # Ensure shading is within valid range
        shading = np.clip(shading, 0.1, 1.0)
        
        # Calculate local texture complexity for adaptive blending
        texture_complexity = np.abs(cv2.Laplacian(luminance_img, cv2.CV_32F, ksize=3))
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
        
        # Create a color layer with the target color
        color_layer = np.zeros_like(source_image, dtype=np.float32)
        for c in range(3):
            color_layer[..., c] = target_bgr[c] * config['color_intensity']
        
        # Apply color with enhanced texture preservation
        for c in range(3):  # BGR channels
            # Calculate adaptive blend factor (higher for complex textures)
            detail_preservation = config['base_detail_preservation'] + (
                config['texture_detail_weight'] * texture_complexity[mask]
            )
            
            # Adjust based on color saturation (more saturated colors need less detail preservation)
            detail_preservation *= (1.0 - (color_saturation * config['saturation_influence']))
            
            # For black t-shirts, we use a different blending approach:
            # 1. Start with the target color
            # 2. Modulate it by the shading map to preserve highlights and shadows
            # 3. Blend in some of the original texture details
            
            # Apply the target color modulated by shading
            colored_image[mask, c] = np.clip(
                color_layer[mask, c] * shading[mask],
                0,
                255
            ).astype(np.uint8)
            
            # Blend in original texture details
            # For dark fabrics, we need to be careful not to bring back too much of the original dark color
            # We'll use a more selective approach to preserve only the texture variations
            
            # Extract texture variations from the original image
            original_texture = source_image[mask, c].astype(np.float32)
            
            # Normalize the texture to center around zero
            mean_texture = np.mean(original_texture)
            centered_texture = original_texture - mean_texture
            
            # Scale the texture variations based on detail preservation
            scaled_texture = centered_texture * detail_preservation * 0.5
            
            # Add the texture variations to the colored image
            colored_image[mask, c] = np.clip(
                colored_image[mask, c].astype(np.float32) + scaled_texture,
                0,
                255
            ).astype(np.uint8)
        
        return colored_image 