import numpy as np
from typing import Optional, Tuple
from src.utils.constants import RIGHT_EYE, LEFT_EYE

def landmark_processor(
        calibrator, 
        results, 
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[Optional[float], str]:
        """
        Optimized landmark processing with reduced allocations.
        
        Returns:
            (ear_value, status_message)
        """
        if not results.multi_face_landmarks:
            return None, "Face not detected"

        if len(results.multi_face_landmarks) > 1:
            return None, "ERROR: Too many faces!"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_shape[:2]

        try:
            # === OPTIMIZATION: Direct indexing without intermediate list ===
            # Right eye
            right_ear = calibrator.ear_calculator.calculate([
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in RIGHT_EYE.ear
            ])
            
            # Left eye
            left_ear = calibrator.ear_calculator.calculate([
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in LEFT_EYE.ear
            ])
            
            ear = (right_ear + left_ear) / 2.0

            # Validate EAR bounds
            if not (calibrator.EAR_BOUNDS[0] < ear < calibrator.EAR_BOUNDS[1]):
                return ear, "Adjust position or lighting"

            # === OPTIMIZATION: Inline stability check without extra arrays ===
            status = "Keep steady..."
            if calibrator._ear_count >= 5:
                # Calculate median of recent samples efficiently
                recent_count = min(calibrator.STABILITY_WINDOW, calibrator._ear_count)
                recent_median = float(np.median(calibrator._ear_buffer[calibrator._ear_count - recent_count:calibrator._ear_count]))
                
                if abs(ear - recent_median) > 0.12:
                    status = "Hold still (stabilizing)"

            # Store in pre-allocated buffer
            if calibrator._ear_count < calibrator.PREALLOCATE_SIZE:
                calibrator._ear_buffer[calibrator._ear_count] = ear
                calibrator._ear_count += 1
            else:
                # Buffer full (shouldn't happen in 10s at 30fps)
                print("Warning: EAR buffer full")

            return ear, f"{status} ({calibrator._ear_count} samples)"

        except (IndexError, ValueError) as e:
            return None, "Could not find eye landmarks"