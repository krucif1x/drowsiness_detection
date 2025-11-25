import numpy as np
from typing import Optional
import cv2

def feedback(
        self, 
        frame: np.ndarray, 
        ear: Optional[float], 
        elapsed: float, 
        status_msg: str, 
        num_samples: int
    ):
        """
        Draw calibration UI with optimized rendering.
        """
        # === OPTIMIZATION: Use class constants to avoid recalculation ===
        bar_length = 300
        bar_height = 20
        margin = 10
        
        # Progress bar
        progress = int(bar_length * min(elapsed / self.CALIBRATION_DURATION_S, 1.0))
        cv2.rectangle(frame, (margin, 60), (margin + bar_length, 60 + bar_height), (200, 200, 200), 2)
        if progress > 0:
            cv2.rectangle(frame, (margin, 60), (margin + progress, 60 + bar_height), (0, 255, 255), -1)

        # Text overlays (cached font for minor speedup)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, f"Calibrating: {int(elapsed)}/{self.CALIBRATION_DURATION_S}s", 
                    (margin, 100), font, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Stable samples: {num_samples}/{self.MIN_VALID_SAMPLES}", 
                    (margin, 130), font, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, "Keep your eyes open and look at the camera", 
                    (margin, 150), font, 0.6, (255, 255, 255), 1)

        # Live EAR value
        if ear is not None:
            color = (0, 255, 0) if self.EAR_BOUNDS[0] < ear < self.EAR_BOUNDS[1] else (0, 165, 255)
            cv2.putText(frame, f"Live EAR: {ear:.3f}", (margin, 30), font, 0.7, color, 2)

        # Status message
        if status_msg:
            msg_color = (0, 0, 255) if "ERROR" in status_msg else (255, 255, 255)
            cv2.putText(frame, status_msg, (margin, 180), font, 0.7, msg_color, 2)

        cv2.imshow("Drowsiness System", frame)
