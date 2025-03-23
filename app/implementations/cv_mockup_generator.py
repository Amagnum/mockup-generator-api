import cv2
import numpy as np
from typing import Dict, Tuple, Any

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
        if source_image is None or source_mask is None or source_depth is None or design_image is None:
            raise ValueError("One or more input images are None")

        # Step 1: Change T-shirt Color
        # Create a copy of the source image
        colored_tshirt_image = source_image.copy()

        # Determine if color is light or dark if in auto mode
        if color_mode == 'auto':
            # Calculate perceived brightness (ITU-R BT.709)
            brightness = 0.2126 * color_code[0] + 0.7152 * color_code[1] + 0.0722 * color_code[2]
            color_mode = 'light' if brightness > 128 else 'dark'

        # Extract mask
        mask = source_mask > 0

        if color_mode == 'light':
            # For light colors, use HSV color space to preserve shading
            # Convert RGB color_code to HSV
            color_rgb = np.array([[color_code]], dtype=np.uint8)
            color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0, 0]

            # Convert source image to HSV and apply color
            hsv_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
            hsv_source[mask, 0] = color_hsv[0]  # Hue
            hsv_source[mask, 1] = color_hsv[1]  # Saturation
            # Optionally adjust value for better light color representation
            # hsv_source[mask, 2] = np.clip(hsv_source[mask, 2] * 0.9, 0, 255)  # Slightly darken
            colored_tshirt_image = cv2.cvtColor(hsv_source, cv2.COLOR_HSV2BGR)
        else:
            # For dark colors, use multiply blending for better results
            # Create a solid color image with the target color
            solid_color = np.zeros_like(source_image)
            solid_color[:] = (color_code[2], color_code[1], color_code[0])  # BGR format

            # Extract lighting information from the original t-shirt
            luminance = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

            # Apply color with lighting preservation
            for c in range(3):  # BGR channels
                # Multiply mode blending: preserves shadows and highlights
                colored_tshirt_image[mask, c] = np.clip(
                    (solid_color[mask, c].astype(float) * luminance[mask] * 1.5),
                    0,
                    255
                ).astype(np.uint8)

        # Step 2: Compute Bounding Box
        dh, dw = design_image.shape[:2]
        scaled_dw = int(dw * scale_factor)
        scaled_dh = int(dh * scale_factor)
        x, y = location
        x_min = int(x - scaled_dw / 2)
        x_max = x_min + scaled_dw
        y_min = int(y - scaled_dh / 2)
        y_max = y_min + scaled_dh

        image_height, image_width = source_image.shape[:2]
        if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            raise ValueError("Design placement exceeds image boundaries.")

        # Step 3: Process 16-bit Depth Map
        D_box = source_depth[y_min:y_max, x_min:x_max].astype(float)
        D_box = (D_box - D_box.min()) / (D_box.max() - D_box.min() + 1e-6)  # Normalize to [0,1]

        # Step 4: Warp Design Using Depth
        k = 1.0  # Bending strength
        Dy, Dx = np.gradient(D_box)
        sx = 1 / (1 + k * np.abs(Dx))
        sy = 1 / (1 + k * np.abs(Dy))
        map_x_box = np.cumsum(sx, axis=1)
        map_y_box = np.cumsum(sy, axis=0)

        total_sx = map_x_box[:, -1]
        total_sy = map_y_box[-1, :]
        total_sx[total_sx == 0] = 1e-6
        total_sy[total_sy == 0] = 1e-6
        scaled_map_x_box = (map_x_box / total_sx[:, np.newaxis]) * (dw - 1)
        scaled_map_y_box = (map_y_box / total_sy[np.newaxis, :]) * (dh - 1)

        map_x = np.full((image_height, image_width), -1, dtype=np.float32)
        map_y = np.full((image_height, image_width), -1, dtype=np.float32)
        map_x[y_min:y_max, x_min:x_max] = scaled_map_x_box
        map_y[y_min:y_max, x_min:x_max] = scaled_map_y_box

        # Warp the 4-channel design (BGRA)
        warped_design = cv2.remap(design_image, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # Step 5: Apply Shading with Factor
        warped_color = warped_design[:, :, :3]  # BGR channels
        warped_alpha = warped_design[:, :, 3].astype(float) / 255  # Alpha channel [0,1]
        shading = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).astype(float) / 255
        # Adjust design brightness based on shading and strength factor
        adjusted_shading = 1 - shading_strength + shading_strength * shading
        design_shaded_color = (warped_color.astype(float) * adjusted_shading[:, :, np.newaxis]).astype(np.uint8)

        # Step 6: Composite with Alpha and Mask
        effective_alpha = warped_alpha * (source_mask.astype(float) / 255)  # Combine alpha with mask
        effective_alpha = effective_alpha[:, :, np.newaxis]  # Shape (h, w, 1)
        final_image = (effective_alpha * design_shaded_color +
                    (1 - effective_alpha) * colored_tshirt_image).astype(np.uint8)

        return final_image 