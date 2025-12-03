import time
import logging
import yaml
from collections import deque

log = logging.getLogger(__name__)

class DistractionDetector:
    """
    Simple camera-appropriate distraction detector:
    - Camera shows face + shoulders only (hands normally NOT visible)
    - If ANY hand appears in frame = driver not holding wheel properly
    - Also checks extreme head angles (looking away)
    """

    def __init__(self, fps=30.0, camera_pitch=0.0, camera_yaw=0.0, config_path="config/detector_config.yaml"):
        self.cfg = self._load_config(config_path, fps)
        self.fps = fps

        # State
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.start_time = None

        # Calibration
        self.cal = {'pitch': camera_pitch, 'yaw': camera_yaw}

        # History: track violations over last N frames
        self.history = deque(maxlen=5)

        # Metrics
        self.metrics = {'total_distractions': 0}

        log.info("DistractionDetector initialized (Simple hand-based)")

    def _load_config(self, path, fps):
        defaults = {
            'yaw_threshold': 40.0,           # degrees - extreme head turn
            'pitch_down_threshold': 28.0,    # degrees - looking way down
            'pitch_up_threshold': 25.0,      # degrees - looking way up
            'time_hands_visible': 1.5,       # seconds with hands visible before unsafe
            'time_gaze': 1.2,                # seconds looking away (reduced from 2.5)
        }
        try:
            with open(path, 'r') as f:
                raw = yaml.safe_load(f) or {}
                dist = raw.get('distraction', {})
                return {
                    'yaw_threshold': dist.get('yaw_threshold', defaults['yaw_threshold']),
                    'pitch_down_threshold': dist.get('pitch_down_threshold', defaults['pitch_down_threshold']),
                    'pitch_up_threshold': dist.get('pitch_up_threshold', defaults['pitch_up_threshold']),
                    'time_hands_visible': dist.get('time_hands_visible', defaults['time_hands_visible']),
                    'time_gaze': dist.get('time_gaze', defaults['time_gaze']),
                }
        except Exception as e:
            log.warning(f"Failed to load distraction config: {e}, using defaults")
            return defaults

    def analyze(self, pitch, yaw, roll, hands=None, face=None):
        """
        Simple logic for face+shoulders camera view:
        1. If ANY hand visible in frame = hands off wheel (unsafe)
        2. If head angle extreme = looking away (unsafe)
        
        Returns (is_distracted, is_new_event)
        """
        if not (abs(pitch) < 90 and abs(yaw) < 90):
            return False, False

        # Check for hands in frame (any hand = distraction)
        hands_visible = hands and len(hands) > 0

        # Check head pose violations
        dp = pitch - self.cal['pitch']
        dy = abs(yaw - self.cal['yaw'])
        
        gaze_violation = (
            dy > self.cfg['yaw_threshold'] or 
            dp > self.cfg['pitch_down_threshold'] or 
            dp < -self.cfg['pitch_up_threshold']
        )

        # Determine violation type and threshold
        violation_type = None
        time_threshold = None

        if hands_visible:
            num_hands = len(hands)
            if num_hands == 1:
                violation_type = "ONE HAND OFF WHEEL"
            else:
                violation_type = "BOTH HANDS OFF WHEEL"
            time_threshold = self.cfg['time_hands_visible']
        elif gaze_violation:
            if dy > self.cfg['yaw_threshold']:
                violation_type = "LOOKING ASIDE"
            elif dp > self.cfg['pitch_down_threshold']:
                violation_type = "LOOKING DOWN"
            else:
                violation_type = "LOOKING UP"
            time_threshold = self.cfg['time_gaze']
        else:
            # No hands visible and good head pose = SAFE (hands on wheel below camera)
            violation_type = None

        # Track in history
        self.history.append(violation_type is not None)

        # Require 3 out of 5 frames to have violation
        history_sum = sum(self.history)
        if history_sum >= 3 and violation_type:
            if self.start_time is None or self.distraction_type != violation_type:
                # New violation or type changed - reset timer
                self.start_time = time.time()
                self.distraction_type = violation_type
            
            elapsed = time.time() - self.start_time
            
            if elapsed >= time_threshold:
                if not self.is_distracted:
                    # New distraction event
                    self.is_distracted = True
                    self.metrics['total_distractions'] += 1
                    return True, True
                else:
                    # Ongoing distraction
                    return True, False
            
            return False, False
        else:
            # No stable violation - reset
            self.start_time = None
            self.is_distracted = False
            self.distraction_type = "NORMAL"
            return False, False

    def get_status(self):
        return {
            'is_distracted': self.is_distracted,
            'type': self.distraction_type,
            'metrics': self.metrics
        }