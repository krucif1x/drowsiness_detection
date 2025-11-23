import time
import logging
import math
import yaml
from collections import deque

log = logging.getLogger(__name__)

class DistractionDetector:
    """
    Focused on distraction only: gaze direction thresholds and hand/phone/eating/wheel context.
    Any fainting detection is handled by FaintingDetector externally.
    """

    def __init__(self, fps=30.0, camera_pitch=0.0, camera_yaw=0.0, config_path="config/detector_config.yaml"):
        self.cfg = self._load_config(config_path, fps)
        self.fps = fps

        # State
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.start_time = None

        # Calibration
        self.cal = {'pitch': camera_pitch, 'yaw': camera_yaw, 'roll': 0.0}

        # History Buffers
        self.history = deque(maxlen=5)

        # Context Trackers (no fainting state here)
        self.context = {'phone': False, 'eating': False, 'wheel': 0, 'drowsy': False, 'eyes_closed': False}
        self.metrics = {'total_distractions': 0}

        log.info("DistractionDetector initialized (Distraction-only)")

    def _load_config(self, path, fps):
        defaults = {
            'yaw': 50.0,
            'pitch_down': 28.0,
            'pitch_up': 25.0,
            'roll': 40.0,
            'phone_min': 18.0,
            'phone_max': 38.0,
            't_gaze': 2.5,
            't_phone': 1.0,
            't_face': 2.0,
            'zone_phone_y': 0.55,
            'zone_phone_x': 0.25,
            'zone_wheel_y': 0.65
        }
        try:
            with open(path, 'r') as f:
                raw = yaml.safe_load(f) or {}
                distraction_cfg = raw.get('distraction', {})
                return {
                    'yaw': distraction_cfg.get('yaw_threshold', defaults['yaw']),
                    'pitch_down': distraction_cfg.get('pitch_down_threshold', defaults['pitch_down']),
                    'pitch_up': distraction_cfg.get('pitch_up_threshold', defaults['pitch_up']),
                    'roll': distraction_cfg.get('roll_threshold', defaults['roll']),
                    'phone_min': distraction_cfg.get('phone_pitch_min', defaults['phone_min']),
                    'phone_max': distraction_cfg.get('phone_pitch_max', defaults['phone_max']),
                    't_gaze': distraction_cfg.get('time_gaze', defaults['t_gaze']),
                    't_phone': distraction_cfg.get('time_phone', defaults['t_phone']),
                    't_face': distraction_cfg.get('time_hand_face', defaults['t_face']),
                    'zone_phone_y': distraction_cfg.get('phone_zone_y_max', defaults['zone_phone_y']),
                    'zone_phone_x': distraction_cfg.get('phone_zone_x_outer', defaults['zone_phone_x']),
                    'zone_wheel_y': distraction_cfg.get('wheel_zone_y_min', defaults['zone_wheel_y'])
                }
        except Exception as e:
            log.warning(f"Failed to load distraction config: {e}")
            return defaults

    def set_drowsiness_state(self, is_drowsy, eyes_closed):
        self.context['drowsy'] = is_drowsy
        self.context['eyes_closed'] = eyes_closed

    def detect_hand_context(self, hands, face):
        self.context.update({'phone': False, 'eating': False, 'wheel': 0})
        if not hands:
            return

        for h in hands:
            wrist, finger = h[0], h[12]
            if wrist[1] > self.cfg['zone_wheel_y']:
                self.context['wheel'] += 1
                continue

            is_high = finger[1] < self.cfg['zone_phone_y']
            is_side = not (self.cfg['zone_phone_x'] < finger[0] < (1.0 - self.cfg['zone_phone_x']))
            if is_high and is_side:
                self.context['phone'] = True
                continue

            if face and ((finger[0]-face[0])**2 + (finger[1]-face[1])**2) ** 0.5 < 0.15:
                if not self.context['phone']:
                    self.context['eating'] = True

    def analyze(self, pitch, yaw, roll, hands=None, face=None):
        """
        Distraction decision only. Fainting must be handled by FaintingDetector separately.
        Returns (is_distracted, is_new_event).
        """
        if not (abs(pitch) < 90 and abs(yaw) < 90):
            return False, False

        self.detect_hand_context(hands, face)

        dp, dy = pitch - self.cal['pitch'], abs(yaw - self.cal['yaw'])
        violations = []
        if self.context['phone']:
            violations.append("PHONE")
        elif dp > self.cfg['pitch_down']:
            violations.append("LOOKING DOWN")
        elif dy > self.cfg['yaw']:
            violations.append("LOOKING ASIDE")
        elif dp < -self.cfg['pitch_up']:
            violations.append("LOOKING UP")
        elif self.context['eating'] and "PHONE" not in violations:
            violations.append("HAND ON FACE")

        self.history.append(len(violations) > 0)
        if sum(self.history) >= 3:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            # Choose timer by type
            if "PHONE" in violations:
                limit = self.cfg['t_phone']
            elif "HAND ON FACE" in violations:
                limit = self.cfg['t_face']
            else:
                limit = self.cfg['t_gaze']

            if elapsed > limit:
                if not self.is_distracted:
                    self.is_distracted = True
                    self.metrics['total_distractions'] += 1
                    self.distraction_type = violations[0] if violations else "DISTRACTED"
                    return True, True
                return True, False
            return False, False
        else:
            self.start_time = None
            self.is_distracted = False
            self.distraction_type = "NORMAL"
            return False, False

    def get_status(self, p, y, r):
        return {
            'is_distracted': self.is_distracted,
            'type': self.distraction_type
        }