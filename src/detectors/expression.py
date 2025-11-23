import math
import logging
import numpy as np
from collections import deque
from typing import List, Tuple

from src.utils.ear.constants import M_MAR

log = logging.getLogger(__name__)

class MouthExpressionClassifier:
    """
    Geometric Classifier for YAWN/SMILE/LAUGH/NEUTRAL.
    Optimized for stability and hand occlusions. Works with:
      - Face landmarks in pixel coordinates [(x,y), ...]
      - Hands data in normalized coordinates (0..1)
    """

    # Smoothing
    EMA_ALPHA = 0.35           # slightly snappier than before
    PERSIST_FRAMES = 6         # a bit more persistence for stability

    # Thresholds
    TH_YAWN_MAR = 0.55
    TH_SMILE_WIDTH_RATIO = 1.15
    TH_SMILE_MAR = 0.25
    TH_LAUGH_MAR = 0.40

    # Hand proximity thresholds (normalized space)
    HAND_FACE_PROX = 0.20      # looser, reduce false OBSCURED
    HAND_MOUTH_PROX = 0.16     # for mouth-specific occlusion

    def __init__(self):
        # Exponential Moving Averages
        self._ema_mar = 0.0
        self._ema_width_ratio = 1.0

        # Label history for stabilization
        self._history = deque(maxlen=self.PERSIST_FRAMES)

        # Baseline neutral mouth width (pixels)
        self._neutral_width = None
        self._frame_count = 0

    @staticmethod
    def _dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)

    def _get_mar(self, lm: List[Tuple[int, int]]) -> float:
        # MAR = (A + B) / (2C)
        A = self._dist(lm[M_MAR[1]], lm[M_MAR[5]])
        B = self._dist(lm[M_MAR[2]], lm[M_MAR[4]])
        C = self._dist(lm[M_MAR[0]], lm[M_MAR[3]])
        if C <= 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def _hand_obscures_mouth(self, landmarks_px: List[Tuple[int, int]], img_w: int, img_h: int, hands_data) -> bool:
        if not hands_data or img_w <= 0 or img_h <= 0:
            return False

        # Mouth center (lip corners)
        lc = landmarks_px[M_MAR[0]]  # left corner
        rc = landmarks_px[M_MAR[3]]  # right corner
        mx_px = 0.5 * (lc[0] + rc[0])
        my_px = 0.5 * (lc[1] + rc[1])

        # Normalize mouth center to compare with hands (0..1)
        mx = mx_px / max(1.0, img_w)
        my = my_px / max(1.0, img_h)

        # Check a couple of fingertip landmarks
        for hand in hands_data:
            # Middle tip (12) and Index tip (8) typical of touching mouth
            for idx in (12, 8):
                try:
                    fx, fy, _ = hand[idx]
                except Exception:
                    continue
                if math.hypot(fx - mx, fy - my) < self.HAND_MOUTH_PROX:
                    return True
        return False

    def classify(self, landmarks: List[Tuple[int, int]], img_h: int, hands_data: list = None) -> str:
        """
        landmarks: face landmarks in pixel coordinates
        img_h: image height (pixels); used only for a quick normalization helper
        hands_data: list of hands; each hand is a list of normalized (x,y,z) landmarks
        """
        # Quick guards
        if not landmarks or len(landmarks) <= max(M_MAR):
            self._history.append("NEUTRAL")
            return self._stable_label()

        self._frame_count += 1

        # Compute features
        try:
            mar = self._get_mar(landmarks)
            width = self._dist(landmarks[M_MAR[0]], landmarks[M_MAR[3]])
        except Exception:
            self._history.append("NEUTRAL")
            return self._stable_label()

        # Calibrate neutral width in first ~30 frames, but freeze after set
        if self._neutral_width is None:
            self._neutral_width = max(width, 1.0)
        elif self._frame_count <= 30:
            # small running max helps avoid very small initial widths
            self._neutral_width = max(self._neutral_width, width)

        width_ratio = width / max(self._neutral_width, 1.0)

        # Update EMAs
        alpha = self.EMA_ALPHA
        self._ema_mar = (1 - alpha) * self._ema_mar + alpha * mar
        self._ema_width_ratio = (1 - alpha) * self._ema_width_ratio + alpha * width_ratio

        # Hand occlusion check (use image width for x, height for y; we only have img_h but can infer an aspect)
        # We do a robust check using mouth center normalized against rough image size.
        # If you can pass img_w too, replace img_w_guess below.
        img_w_guess = img_h  # assume near-square FOV; safe default
        if self._hand_obscures_mouth(landmarks, img_w_guess, img_h, hands_data):
            self._history.append("OBSCURED")
            return self._stable_label()

        # Classification logic using smoothed features
        curr_mar = self._ema_mar
        curr_width = self._ema_width_ratio

        label = "NEUTRAL"
        if curr_mar > self.TH_YAWN_MAR:
            label = "YAWN"
        elif curr_mar > self.TH_LAUGH_MAR and curr_width > 1.10:
            label = "LAUGH"
        elif curr_width > self.TH_SMILE_WIDTH_RATIO and curr_mar < self.TH_SMILE_MAR:
            label = "SMILE"

        self._history.append(label)
        return self._stable_label()

    def _stable_label(self) -> str:
        if not self._history:
            return "NEUTRAL"
        # Majority vote
        unique, counts = np.unique(self._history, return_counts=True)
        return str(unique[int(np.argmax(counts))])