import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional
from app.utils.logging_config import get_logger
from app.utils.debug_utils import debug_timing, debug_exception, save_debug_image

class ColorHandlerV3:
    """
    Advanced color transformation handler for realistic t-shirt recoloring
    
    Implements color transformation techniques to recolor garments while
    preserving original lighting, shadows, and fabric details.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @debug_timing
    @debug_exception
    def apply_tshirt_color(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int],
        color_mode: str = 'auto',
        method: str = 'advanced',
        config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Apply color transformation to the t-shirt area in the source image.
        
        Parameters:
        -----------
        source_image : np.ndarray
            BGR image of a person wearing a t-shirt
        mask : np.ndarray
            Binary mask where 255 indicates t-shirt area and 0 elsewhere
        color_code : Tuple[int, int, int]
            Target color in RGB format (e.g., (255, 0, 0) for red)
        color_mode : str
            Mode for color application, one of 'auto', 'light', 'dark'
        method : str
            Color transformation method to use
        config : Dict[str, Any], optional
            Additional configuration parameters for fine-tuning
            
        Returns:
        --------
        np.ndarray
            BGR image with the t-shirt recolored
        """
        if config is None:
            config = {}
            
        # Default configuration parameters with fallbacks
        brightness_preserve = config.get('brightness_preserve', 0.7)
        saturation_boost = config.get('saturation_boost', 1.1)
        detail_enhance = config.get('detail_enhance', 0.7)
        color_shift = config.get('color_shift', 0)
        smooth_transition = config.get('smooth_transition', True)
        
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create normalized mask for blending (0.0 - 1.0)
        norm_mask = binary_mask.astype(float) / 255.0
        if len(norm_mask.shape) == 2:
            norm_mask = np.expand_dims(norm_mask, axis=-1)
        
        # Analyze source image brightness to determine if it's a dark or light garment
        source_brightness = self._analyze_source_brightness(source_image, binary_mask)
        
        # Analyze target color brightness
        target_brightness = 0.299 * color_code[0] + 0.587 * color_code[1] + 0.114 * color_code[2]
        
        # Log brightness values for debugging
        self.logger.debug(f"Source brightness: {source_brightness}, Target brightness: {target_brightness}")
        
        # Choose method based on input
        if method == 'basic':
            return self._basic_color_replacement(source_image, binary_mask, color_code)
        elif method == 'luminance':
            return self._luminance_preserving_coloring(source_image, binary_mask, color_code, 
                                                     brightness_preserve, saturation_boost)
        elif method == 'extreme_transform' or (source_brightness < 50 and target_brightness > 200):
            # Use extreme transform for very dark to very light transformations
            self.logger.info("Using extreme transform method for dark-to-light conversion")
            return self._extreme_transformation(source_image, binary_mask, color_code, detail_enhance, smooth_transition)
        elif method == 'advanced':
            # Dynamically adjust parameters for dark to light transformations
            if source_brightness < 100 and target_brightness > 150:
                # Reduce brightness preservation for dark to light
                brightness_preserve = min(brightness_preserve, 0.4)
                detail_enhance = min(detail_enhance, 0.5)
                self.logger.debug(f"Adjusted parameters for dark-to-light: brightness_preserve={brightness_preserve}, detail_enhance={detail_enhance}")
            
            return self._advanced_coloring(source_image, binary_mask, color_code, color_mode, 
                                         brightness_preserve, saturation_boost, detail_enhance, 
                                         color_shift, smooth_transition)
        else:
            self.logger.warning(f"Unknown coloring method '{method}', falling back to advanced")
            return self._advanced_coloring(source_image, binary_mask, color_code, color_mode,
                                         brightness_preserve, saturation_boost, detail_enhance,
                                         color_shift, smooth_transition)
    
    def _analyze_source_brightness(self, source_image: np.ndarray, mask: np.ndarray) -> float:
        """Analyze the brightness of the source image in the masked area"""
        # Convert to grayscale for brightness analysis
        if len(source_image.shape) == 3:
            gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = source_image
            
        # Create mask for analysis
        mask_bool = mask > 0
        
        # If no pixels in mask, return 0
        if not np.any(mask_bool):
            return 0
            
        # Calculate mean brightness of masked area
        masked_pixels = gray[mask_bool]
        mean_brightness = np.mean(masked_pixels)
        
        return mean_brightness
    
    def _basic_color_replacement(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int]
    ) -> np.ndarray:
        """Simple direct color replacement with minimal preservation of original details"""
        # Create solid color image with target color (BGR format)
        color_bgr = (color_code[2], color_code[1], color_code[0])  # RGB to BGR
        colored_image = np.zeros_like(source_image)
        colored_image[:] = color_bgr
        
        # Create mask for blending
        normalized_mask = mask.astype(float) / 255.0
        if len(normalized_mask.shape) == 2:
            normalized_mask = np.expand_dims(normalized_mask, axis=-1)
        
        # Blend the colored image with the original using the mask
        result = source_image.copy()
        result = (1 - normalized_mask) * result + normalized_mask * colored_image
        
        return result.astype(np.uint8)
    
    def _extreme_transformation(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int],
        detail_preserve: float = 0.3,
        smooth_transition: bool = True
    ) -> np.ndarray:
        """
        Special method for extreme transformations like black to white
        with minimal reliance on source brightness
        """
        # Convert the source image to grayscale for detail extraction
        gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        
        # Extract texture details using edge detection or local contrast
        # Use a combination of methods for better detail extraction from dark fabrics
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        detail_map = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Enhance details with local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_details = cv2.Laplacian(enhanced_gray, cv2.CV_64F)
        
        # Combine detail maps
        detail_map = (detail_map + np.abs(enhanced_details)) / 2
        detail_map = cv2.normalize(detail_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Create base color image (solid color) - handle different channel counts
        color_bgr = (color_code[2], color_code[1], color_code[0])  # RGB to BGR
        colored_image = np.zeros_like(source_image)
        
        # Check the number of channels in the source image
        if source_image.shape[2] == 4:  # BGRA
            # For BGRA images, we need to provide all 4 channels
            color_bgra = (color_code[2], color_code[1], color_code[0], 255)  # RGB to BGRA with full alpha
            colored_image[:] = color_bgra
        else:  # BGR
            colored_image[:] = color_bgr
        
        # Prepare mask for blending
        normalized_mask = mask.astype(float) / 255.0
        if len(normalized_mask.shape) == 2:
            normalized_mask = np.expand_dims(normalized_mask, axis=-1)
        
        # Apply smooth transitions if requested
        if smooth_transition:
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)
            transition_mask = mask - mask_eroded
            transition_mask_norm = transition_mask.astype(float) / 255.0
            if len(transition_mask_norm.shape) == 2:
                transition_mask_norm = np.expand_dims(transition_mask_norm, axis=-1)
                
            # Create a gradient mask for smooth transitions
            effective_mask = normalized_mask - (transition_mask_norm * 0.8)
        else:
            effective_mask = normalized_mask
        
        # Scale detail map for subtler effect on light colors
        weighted_detail = detail_map * detail_preserve * 30
        weighted_detail = np.clip(weighted_detail, 0, 60)
        
        # Convert detail map to 3D if needed
        if len(weighted_detail.shape) == 2:
            weighted_detail = np.expand_dims(weighted_detail, axis=-1)
        
        # Apply details to the colored image
        # Darker for details (subtract), brighter for highlights (add)
        target_brightness = 0.299 * color_code[0] + 0.587 * color_code[1] + 0.114 * color_code[2]
        
        if target_brightness > 200:  # Very light target color
            # For light colors, details should darken
            detail_effect = -weighted_detail
        elif target_brightness < 50:  # Very dark target color
            # For dark colors, details should lighten
            detail_effect = weighted_detail
        else:
            # For mid-range colors, use a mix
            detail_effect = weighted_detail * (0.5 - target_brightness/255)
        
        # Ensure the detail effect is properly broadcasted to all channels
        # For BGRA images, we need to ensure alpha is not affected
        if source_image.shape[2] == 4:  # BGRA
            # Create a 4-channel detail effect with zeros for alpha channel
            detail_effect_bgra = np.zeros((detail_effect.shape[0], detail_effect.shape[1], 4), dtype=np.float32)
            detail_effect_bgra[:, :, :3] = detail_effect  # Only apply to BGR channels
            detail_effect = detail_effect_bgra
            
        # Apply detail effects to the solid color
        colored_with_details = colored_image.copy().astype(np.float32)
        colored_with_details += detail_effect
        colored_with_details = np.clip(colored_with_details, 0, 255).astype(np.uint8)
        
        # Final composite
        result = source_image.copy()
        
        # For BGRA images, handle blending carefully to preserve alpha
        if source_image.shape[2] == 4:
            # Blend BGR channels with the mask
            for i in range(3):  # Only BGR channels, not alpha
                result[:, :, i] = (1 - effective_mask[:, :, 0]) * result[:, :, i] + effective_mask[:, :, 0] * colored_with_details[:, :, i]
        else:
            # Regular blending for BGR images
            result = (1 - effective_mask) * result + effective_mask * colored_with_details
        
        return result.astype(np.uint8)
    
    def _luminance_preserving_coloring(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int],
        brightness_preserve: float = 0.7,
        saturation_boost: float = 1.1
    ) -> np.ndarray:
        """
        Preserves the luminance of the original image while changing the hue and saturation
        """
        # Convert the BGR image to LAB color space
        lab_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
        
        # Create target color in LAB space
        target_rgb = np.array([[color_code]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)[0, 0]
        
        # Extract L, A, B channels
        L, A, B = cv2.split(lab_image)
        
        # Apply mask to the A and B channels (chromaticity)
        mask_3d = mask / 255.0
        if len(mask_3d.shape) == 2:
            mask_3d = np.expand_dims(mask_3d, axis=2)
        
        # Calculate mean brightness of source and target
        source_brightness = np.mean(L[mask > 0]) if np.any(mask > 0) else 128
        target_brightness = target_lab[0]
        
        # Adjust brightness preservation based on the transformation direction
        adjusted_brightness = brightness_preserve
        if source_brightness < 100 and target_brightness > 180:
            # For dark to light transformation, reduce preservation
            adjusted_brightness = max(0.3, brightness_preserve - 0.4)
            self.logger.debug(f"Adjusted brightness preservation to {adjusted_brightness} for dark-to-light")
        
        # Scale luminance to preserve brightness based on target color
        L_masked = L * mask_3d[:, :, 0]
        
        # For dark-to-light transformations, use a more aggressive approach
        if source_brightness < 100 and target_brightness > 180:
            # Create a stretched and centered version of original luminance
            L_adjusted = (L_masked - source_brightness) * (255 / max(1, source_brightness * 2))
            L_adjusted = np.clip(L_adjusted + target_brightness, 0, 255)
            L_scaled = np.where(mask_3d[:, :, 0] > 0, L_adjusted, L)
        else:
            # Normal scaling approach
            L_scaled = np.where(mask_3d[:, :, 0] > 0, 
                              L_masked * adjusted_brightness + (1 - adjusted_brightness) * target_lab[0],
                              L)
        
        # Create new A and B channels based on target color
        A_new = np.where(mask > 0, target_lab[1], A)
        B_new = np.where(mask > 0, target_lab[2], B)
        
        # Merge channels back
        colored_lab = cv2.merge([L_scaled.astype(np.uint8), 
                                A_new.astype(np.uint8), 
                                B_new.astype(np.uint8)])
        
        # Convert back to BGR
        colored_image = cv2.cvtColor(colored_lab, cv2.COLOR_LAB2BGR)
        
        return colored_image
    
    def _advanced_coloring(
        self,
        source_image: np.ndarray,
        mask: np.ndarray,
        color_code: Tuple[int, int, int],
        color_mode: str = 'auto',
        brightness_preserve: float = 0.7,
        saturation_boost: float = 1.1,
        detail_enhance: float = 0.7,
        color_shift: int = 0,
        smooth_transition: bool = True
    ) -> np.ndarray:
        """
        Advanced coloring technique that preserves lighting, fabric details and
        applies realistic color transformations
        """
        # Convert the source image to multiple color spaces for processing
        hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
        lab_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
        
        # Calculate source brightness (for dark/light fabric detection)
        source_brightness = self._analyze_source_brightness(source_image, mask)
        
        # Determine if target color is light or dark
        target_brightness = 0.299 * color_code[0] + 0.587 * color_code[1] + 0.114 * color_code[2]
        
        # Override color_mode if auto and set based on target brightness
        if color_mode == 'auto':
            color_mode = 'light' if target_brightness > 127 else 'dark'
        
        # Create target color in HSV and LAB spaces
        target_rgb = np.array([[color_code]], dtype=np.uint8)
        target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)[0, 0]
        
        # Special handling for dark-to-light transformations
        is_dark_to_light = source_brightness < 80 and target_brightness > 180
        
        # Process based on the color mode
        if is_dark_to_light:
            # For dramatic dark to light transformations, use a specialized approach
            self.logger.debug("Using specialized dark-to-light transformation")
            return self._process_dark_to_light(source_image, hsv_image, lab_image, mask,
                                             target_hsv, target_lab, brightness_preserve,
                                             saturation_boost, detail_enhance, color_shift,
                                             smooth_transition)
        elif color_mode == 'dark':
            return self._process_dark_garment(source_image, hsv_image, lab_image, mask, 
                                             target_hsv, target_lab, brightness_preserve,
                                             saturation_boost, detail_enhance, color_shift,
                                             smooth_transition)
        else:  # 'light' mode
            return self._process_light_garment(source_image, hsv_image, lab_image, mask, 
                                              target_hsv, target_lab, brightness_preserve,
                                              saturation_boost, detail_enhance, color_shift,
                                              smooth_transition)
    
    def _process_dark_to_light(
        self,
        source_image: np.ndarray,
        hsv_image: np.ndarray,
        lab_image: np.ndarray,
        mask: np.ndarray,
        target_hsv: np.ndarray,
        target_lab: np.ndarray,
        brightness_preserve: float,
        saturation_boost: float,
        detail_enhance: float,
        color_shift: int,
        smooth_transition: bool
    ) -> np.ndarray:
        """Specialized processing for dark garments being transformed to light colors"""
        # For dark to light, we'll extract details more aggressively and use minimal brightness preservation
        
        # Extract channels
        l, a, b = cv2.split(lab_image)
        h, s, v = cv2.split(hsv_image)
        
        # Use CLAHE to enhance contrast and extract details from dark areas
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Create a detail map using multiple techniques
        detail_map1 = cv2.Laplacian(enhanced_l, cv2.CV_64F)
        sobel_x = cv2.Sobel(enhanced_l, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced_l, cv2.CV_64F, 0, 1, ksize=3)
        detail_map2 = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Combine detail maps
        detail_map = (np.abs(detail_map1) + detail_map2) / 2
        detail_map = cv2.normalize(detail_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply strong detail enhancement but inverted (dark details on light background)
        detail_factor = detail_enhance * 25  # Stronger effect
        
        # Adjust mask for smooth transitions if requested
        if smooth_transition:
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)
            transition_mask = mask - mask_eroded
            transition_mask_norm = transition_mask.astype(float) / 255.0
        
        # Normalize mask
        mask_norm = mask.astype(float) / 255.0
        
        # For dark to light transformation:
        # 1. Start with target color
        # 2. Apply inverted details
        # 3. Blend minimally with original
        
        # Create new LAB image starting with target color
        l_new = np.full_like(l, target_lab[0], dtype=np.float32)
        a_new = np.full_like(a, target_lab[1], dtype=np.float32)
        b_new = np.full_like(b, target_lab[2], dtype=np.float32)
        
        # Apply inverted details to luminance
        l_with_details = l_new - (detail_map * detail_factor) 
        l_with_details = np.clip(l_with_details, 0, 255)
        
        # Apply the mask
        l_final = np.where(mask > 0, l_with_details, l)
        a_final = np.where(mask > 0, a_new, a)
        b_final = np.where(mask > 0, b_new, b)
        
        # Handle smooth transitions if enabled
        if smooth_transition:
            l_final = l_final * (mask_norm - transition_mask_norm * 0.5) + l * transition_mask_norm * 0.5
            a_final = a_final * (mask_norm - transition_mask_norm * 0.5) + a * transition_mask_norm * 0.5
            b_final = b_final * (mask_norm - transition_mask_norm * 0.5) + b * transition_mask_norm * 0.5
        
        # Merge LAB channels
        lab_new = cv2.merge([l_final.astype(np.uint8), 
                           a_final.astype(np.uint8), 
                           b_final.astype(np.uint8)])
        
        # Convert back to BGR
        return cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
    
    def _process_dark_garment(
        self,
        source_image: np.ndarray,
        hsv_image: np.ndarray,
        lab_image: np.ndarray,
        mask: np.ndarray,
        target_hsv: np.ndarray,
        target_lab: np.ndarray,
        brightness_preserve: float,
        saturation_boost: float,
        detail_enhance: float,
        color_shift: int,
        smooth_transition: bool
    ) -> np.ndarray:
        """Process a dark garment for recoloring to another dark/medium color"""
        # Extract channels
        h, s, v = cv2.split(hsv_image)
        l, a, b = cv2.split(lab_image)
        
        # Create a detail map from the original image
        gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        detail_map = cv2.Laplacian(gray, cv2.CV_64F)
        detail_map = np.abs(detail_map)
        detail_map = cv2.normalize(detail_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Adjust the mask for smooth transitions if requested
        if smooth_transition:
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)
            transition_mask = mask - mask_eroded
            transition_mask_norm = transition_mask.astype(float) / 255.0
        
        # Normalize mask to 0.0-1.0 range
        mask_norm = mask.astype(float) / 255.0
        
        # Create new HSV image
        h_new = np.where(mask > 0, target_hsv[0] + color_shift, h)
        
        # For dark garments, boost saturation
        s_boost = min(255, target_hsv[1] * saturation_boost)
        s_new = np.where(mask > 0, s_boost, s)
        
        # Value (brightness) transformation preserving original lighting
        v_preserved = v * brightness_preserve
        v_new = np.where(mask > 0, 
                         np.clip(v_preserved + (1-brightness_preserve) * target_hsv[2], 0, 255),
                         v)
        
        # Add back details using the detail map
        if detail_enhance > 0:
            detail_contribution = detail_map * detail_enhance * 20
            v_new = np.where(mask > 0, 
                            np.clip(v_new - detail_contribution, 0, 255),
                            v_new)
        
        # Handle smooth transitions if enabled
        if smooth_transition:
            h_new = h_new * (mask_norm - transition_mask_norm * 0.5) + h * transition_mask_norm * 0.5
            s_new = s_new * (mask_norm - transition_mask_norm * 0.5) + s * transition_mask_norm * 0.5
            v_new = v_new * (mask_norm - transition_mask_norm * 0.5) + v * transition_mask_norm * 0.5
        
        # Merge channels and convert back to BGR
        hsv_new = cv2.merge([h_new.astype(np.uint8), 
                            s_new.astype(np.uint8), 
                            v_new.astype(np.uint8)])
        
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    
    def _process_light_garment(
        self,
        source_image: np.ndarray,
        hsv_image: np.ndarray,
        lab_image: np.ndarray,
        mask: np.ndarray,
        target_hsv: np.ndarray,
        target_lab: np.ndarray,
        brightness_preserve: float,
        saturation_boost: float,
        detail_enhance: float,
        color_shift: int,
        smooth_transition: bool
    ) -> np.ndarray:
        """Process a light garment for recoloring"""
        # For light garments, we'll use LAB color space for better results
        l, a, b = cv2.split(lab_image)
        
        # Create a detail map to preserve fabric texture
        gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        detail_map = cv2.Laplacian(gray, cv2.CV_64F)
        detail_map = np.abs(detail_map)
        detail_map = cv2.normalize(detail_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Adjust mask for smooth transitions if requested
        if smooth_transition:
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)
            transition_mask = mask - mask_eroded
            transition_mask_norm = transition_mask.astype(float) / 255.0
        
        # Normalize mask
        mask_norm = mask.astype(float) / 255.0
        
        # Apply the target color while preserving luminance variations
        l_preserved = l * brightness_preserve
        l_new = np.where(mask > 0, 
                         np.clip(l_preserved + (1-brightness_preserve) * target_lab[0], 0, 255),
                         l)
        
        # Add details back
        if detail_enhance > 0:
            detail_contribution = detail_map * detail_enhance * 20
            l_new = np.where(mask > 0, 
                            np.clip(l_new - detail_contribution, 0, 255),
                            l_new)
        
        # Set A and B channels to target color
        a_new = np.where(mask > 0, target_lab[1], a)
        b_new = np.where(mask > 0, target_lab[2], b)
        
        # Handle smooth transitions if enabled
        if smooth_transition:
            l_new = l_new * (mask_norm - transition_mask_norm * 0.5) + l * transition_mask_norm * 0.5
            a_new = a_new * (mask_norm - transition_mask_norm * 0.5) + a * transition_mask_norm * 0.5
            b_new = b_new * (mask_norm - transition_mask_norm * 0.5) + b * transition_mask_norm * 0.5
        
        # Merge channels and convert back to BGR
        lab_new = cv2.merge([l_new.astype(np.uint8), 
                            a_new.astype(np.uint8), 
                            b_new.astype(np.uint8)])
        
        return cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR) 