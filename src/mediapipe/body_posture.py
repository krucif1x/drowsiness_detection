import cv2
import numpy as np
import logging
from mediapipe.python.solutions import pose

from src.utils.constants import UPPER_BODY_LANDMARKS, UpperBodyIdx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UpperBodyPose")


class MediapipeUpperBodyModel:

    def __init__(self):
        # Lightweight settings for in-cabin driver camera
        self.static_image_mode = False
        self.model_complexity = 1
        self.smooth_landmarks = True
        self.enable_segmentation = False
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5

        logger.info("Initializing Upper-Body Mediapipe Pose model...")
        self.load_model()

    # ---------------------------------------------------------------------

    def load_model(self):

        self.pose = pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        logger.info("Model loaded successfully.")
        
    def inference(self, image: np.ndarray, preprocessed=False):
        """Return only upper-body landmarks as a dict."""
        
        if not preprocessed:
            image = self.preprocess(image)

        results = self.pose.process(image)

        if not results.pose_landmarks:
            return {}

        lm = results.pose_landmarks.landmark

        # Example output:
        # { 11: (x, y, z, visibility), 12: (...), ... }
        output = {}

        for idx in UPPER_BODY_LANDMARKS:
            l = lm[idx]
            output[idx] = (l.x, l.y, l.z, l.visibility)

        return output

    # ---------------------------------------------------------------------

    def get(self, landmarks, idx):
        """Convenience helper."""
        return landmarks.get(idx, None)
