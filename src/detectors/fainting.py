import time
import logging
import math
import yaml
import numpy as np
from collections import deque

log = logging.getLogger(__name__)

class FaintingDetector:
    """
    Dedicated fainting detector using head pose dynamics, hand/face context, and drowsiness state.
    Inputs are lightweight (pitch/yaw/roll, hands, face, context), so upstream modules can share it.
    """

    def __init__(self, fps: float = 30.0, camera_pitch: float = 0.0, config_path: str = "config/detector_config.yaml"):
        self.fps = fps
        self.cfg = self._load_config(config_path, fps)
        # Calibration
        self.cal = {'pitch': camera_pitch}

        # State and metrics
        self.is_fainting = False
        self.faint_probability = 0.0
        self.metrics = {'extreme_pitch_frames': 0, 'pitch_stable_frames': 0, 'total_faints': 0}

        # Histories
        self.pitch_history = deque(maxlen=self.cfg['history_frames'])
        self.face_y_history = deque(maxlen=int(1.0 * fps))

        # Context from other detectors/pipeline
        # These flags are set via set_context() per frame
        self.context = {
            'drowsy': False,        # from drowsiness
            'eyes_closed': False,   # from drowsiness or eye module
            'phone': False,         # from distraction/hand context
            'wheel': 0,             # count of wrists near wheel zone
        }

        log.info("FaintingDetector initialized")

    def _load_config(self, path: str, fps: float):
        defaults = {
            'pitch_extreme': 45.0,
            'rapid_change': 12.0,
            'history_sec': 0.5,
            'phone_pitch_min': 18.0,
            'phone_pitch_max': 38.0,
            'var_stable_max': 5.0,
            'var_unstable_min': 10.0,
            'extreme_frames_min': 5,
            'face_slump_trend_min': 0.005,  # slope threshold
            'prob_threshold': 0.6
        }
        try:
            with open(path, 'r') as f:
                raw = yaml.safe_load(f) or {}
                faint = raw.get('fainting', {})
                return {
                    'pitch_extreme': faint.get('pitch_extreme', defaults['pitch_extreme']),
                    'rapid_change': faint.get('rapid_change', defaults['rapid_change']),
                    'history_frames': int(faint.get('history_sec', defaults['history_sec']) * fps),
                    'phone_pitch_min': faint.get('phone_pitch_min', defaults['phone_pitch_min']),
                    'phone_pitch_max': faint.get('phone_pitch_max', defaults['phone_pitch_max']),
                    'var_stable_max': faint.get('var_stable_max', defaults['var_stable_max']),
                    'var_unstable_min': faint.get('var_unstable_min', defaults['var_unstable_min']),
                    'extreme_frames_min': faint.get('extreme_frames_min', defaults['extreme_frames_min']),
                    'face_slump_trend_min': faint.get('face_slump_trend_min', defaults['face_slump_trend_min']),
                    'prob_threshold': faint.get('prob_threshold', defaults['prob_threshold']),
                }
        except Exception as e:
            log.warning(f"Failed to load fainting config: {e}")
            return {
                'pitch_extreme': defaults['pitch_extreme'],
                'rapid_change': defaults['rapid_change'],
                'history_frames': int(defaults['history_sec'] * fps),
                'phone_pitch_min': defaults['phone_pitch_min'],
                'phone_pitch_max': defaults['phone_pitch_max'],
                'var_stable_max': defaults['var_stable_max'],
                'var_unstable_min': defaults['var_unstable_min'],
                'extreme_frames_min': defaults['extreme_frames_min'],
                'face_slump_trend_min': defaults['face_slump_trend_min'],
                'prob_threshold': defaults['prob_threshold'],
            }

    def set_context(self, drowsy: bool, eyes_closed: bool, phone: bool, wheel_count: int):
        self.context['drowsy'] = drowsy
        self.context['eyes_closed'] = eyes_closed
        self.context['phone'] = phone
        self.context['wheel'] = wheel_count

    def _indicators(self, pitch: float, face_center):
        inds = {'rapid': False, 'extreme': False, 'unstable': False, 'slump': False, 'limp': False, 'phone_pose': False, 'was_drowsy': False}
        self.pitch_history.append(pitch)
        if face_center:
            self.face_y_history.append(face_center[1])

        # Rapid pitch drop over history window
        if len(self.pitch_history) >= max(2, self.cfg['history_frames']):
            if (pitch - self.pitch_history[0]) > self.cfg['rapid_change']:
                inds['rapid'] = True

        # Extreme pitch relative to calibration
        delta = pitch - self.cal['pitch']
        if delta > self.cfg['pitch_extreme']:
            inds['extreme'] = True
            self.metrics['extreme_pitch_frames'] += 1
        else:
            self.metrics['extreme_pitch_frames'] = 0

        # Stability and phone-like pose
        if len(self.pitch_history) >= 10:
            segment = list(self.pitch_history)[-10:]
            var = np.var(segment)
            if self.cfg['phone_pitch_min'] < delta < self.cfg['phone_pitch_max'] and var < self.cfg['var_stable_max']:
                inds['phone_pose'] = True
                self.metrics['pitch_stable_frames'] += 1
            else:
                self.metrics['pitch_stable_frames'] = 0

            if var > self.cfg['var_unstable_min'] or self.metrics['extreme_pitch_frames'] > self.cfg['extreme_frames_min']:
                inds['unstable'] = True

        # Slumping: face y trending downward
        if len(self.face_y_history) >= 20:
            ys = list(self.face_y_history)[-20:]
            slope = np.polyfit(range(len(ys)), ys, 1)[0]
            if slope > self.cfg['face_slump_trend_min']:
                inds['slump'] = True

        # Limp: no phone and no hands at wheel
        if not self.context['phone'] and self.context['wheel'] == 0:
            inds['limp'] = True

        # Prior drowsiness increases risk
        if self.context['drowsy']:
            inds['was_drowsy'] = True

        return inds

    def _score(self, inds):
        score = 0.0
        if inds['rapid'] and self.context['eyes_closed']:
            score += 0.35
        if inds['extreme'] and inds['slump']:
            score += 0.30
        if inds['unstable'] and inds['limp']:
            score += 0.25
        if inds['was_drowsy'] and inds['rapid']:
            score += 0.20

        # Demotions for plausible phone posture
        if inds['phone_pose'] and self.context['phone']:
            score -= 0.40
        if self.context['phone'] and not self.context['eyes_closed']:
            score -= 0.30

        return max(0.0, min(1.0, score))

    def analyze(self, pitch: float, yaw: float, roll: float, hands=None, face_center=None):
        """
        Returns (is_fainting, is_new_event) and updates faint_probability.
        Note: we do not perform hand context classification here beyond wheel/phone flags,
        since that should be provided via set_context by the pipeline or a Hand/Distraction module.
        """
        # Basic sanity check on angles
        if not (abs(pitch) < 90 and abs(yaw) < 90):
            return False, False

        inds = self._indicators(pitch, face_center)
        self.faint_probability = self._score(inds)

        if self.faint_probability > self.cfg['prob_threshold']:
            if not self.is_fainting:
                self.is_fainting = True
                self.metrics['total_faints'] += 1
                log.critical(f"FAINT DETECTED (prob={self.faint_probability:.2f})")
                return True, True
            return True, False
        else:
            self.is_fainting = False
            return False, False

    def get_status(self):
        return {
            'is_fainting': self.is_fainting,
            'faint_prob': self.faint_probability,
            'metrics': self.metrics
        }