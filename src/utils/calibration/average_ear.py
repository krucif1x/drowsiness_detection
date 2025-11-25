import numpy as np
from typing import Optional, Union, Tuple

def average_ear(calibrator) -> Optional[float]:
        """
        Compute robust average using MAD filtering.
        Then calculate threshold as 75% of open-eye baseline.
        """
        if calibrator._ear_count < calibrator.MIN_VALID_SAMPLES:
            print(f"Calibration failed: Not enough stable eye readings ({calibrator._ear_count}/{calibrator.MIN_VALID_SAMPLES}).")
            return None

        # Use only filled portion of buffer
        arr = calibrator._ear_buffer[:calibrator._ear_count]
        
        # Vectorized MAD calculation
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))

        # Keep samples within ~3 sigma equivalent
        if mad < 1e-6:
            filtered = arr  # All samples are nearly identical
        else:
            sigma = 1.4826 * mad
            mask = np.abs(arr - median) <= 3.0 * sigma
            filtered = arr[mask]

        # Fallback: 10-90% trimmed range if MAD filtering too aggressive
        if filtered.size < calibrator.MIN_VALID_SAMPLES:
            lo, hi = np.percentile(arr, [10, 90])
            filtered = arr[(arr >= lo) & (arr <= hi)]

        # Final fallback: use all samples
        if filtered.size < calibrator.MIN_VALID_SAMPLES:
            filtered = arr

        # === FIX: Calculate baseline (open eyes) and threshold ===
        open_eye_baseline = float(np.mean(filtered))
        
        # Threshold is 75% of open-eye baseline
        # (i.e., eyes must close to 75% of normal open position to trigger alert)
        ear_threshold = open_eye_baseline * 0.75
        
        # Log statistics
        print(f"Calibration complete:")
        print(f"  - Total samples: {calibrator._ear_count}")
        print(f"  - Used samples: {filtered.size}")
        print(f"  - Open-eye baseline: {open_eye_baseline:.3f}")
        print(f"  - EAR threshold (75%): {ear_threshold:.3f}")  # ← NEW
        print(f"  - EAR range: {float(np.min(filtered)):.3f} - {float(np.max(filtered)):.3f}")
        
        return ear_threshold