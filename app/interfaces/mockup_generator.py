from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np

class MockupGenerator(ABC):
    """Interface for mockup generation implementations"""
    
    @abstractmethod
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
        Generate a t-shirt mockup with a design placed at a 2D location, bent using a depth map.
        
        Returns:
            np.ndarray: The final mockup image
        """
        pass 