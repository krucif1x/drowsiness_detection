from abc import ABC, abstractmethod
from math import dist
from typing import List, Tuple

class AspectRatio(ABC):
    @abstractmethod
    def calculate(self, landmarks: List[Tuple[float, float]]) -> float:
        pass

class EAR(AspectRatio):
    def calculate(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio.
        Expects 6 points: [Corner1, Top1, Top2, Corner2, Bot2, Bot1]
        """
        # Vertical distances
        A = dist(landmarks[1], landmarks[5])
        B = dist(landmarks[2], landmarks[4])
        
        # Horizontal distance
        C = dist(landmarks[0], landmarks[3])
        
        # SAFETY: Prevent division by zero if eye is fully closed
        if C < 1e-6:
            return 0.0
        
        return (A + B) / (2.0 * C)
    
class MAR(AspectRatio):
    def calculate(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio.
        """
        # NOTE: Ensure M_MAR in constants.py matches this circular order:
        # [LeftCorner, Upper1, Upper2, RightCorner, Lower2, Lower1]
        
        A = dist(landmarks[1], landmarks[5]) 
        B = dist(landmarks[2], landmarks[4]) 
        C = dist(landmarks[0], landmarks[3]) 
        
        if C < 1e-6:
            return 0.0
        
        return (A + B) / (2.0 * C)