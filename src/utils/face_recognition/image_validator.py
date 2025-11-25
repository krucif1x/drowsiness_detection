import cv2
import numpy as np
from functools import lru_cache

class ImageValidator:
    def __init__(self, input_color: str = "RGB"):
        self.input_color = input_color.upper()

    @lru_cache(maxsize=128)
    def _shape_ok(self, shape: tuple, dtype_str: str) -> bool:
        return len(shape) == 3 and shape[2] == 3

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.input_color == "BGR":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be np.ndarray")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not self._shape_ok(frame.shape, str(frame.dtype)):
            raise ValueError(f"Invalid image shape: {frame.shape}")
        return frame