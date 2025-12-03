import cv2
import numpy as np
from mediapipe.python.solutions import hands

class MediaPipeHandsWrapper:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.model = hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def infer(self, image: np.ndarray, preprocessed: bool = False):
        if not preprocessed:
            image = self.preprocess(image)
        result = self.model.process(image)
        
        hands_data = []
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                hands_data.append([
                    (lm.x, lm.y, lm.z) for lm in hand_landmark.landmark
                ])
        return hands_data

    def get_landmark(self, single_hand_data, landmark_index: int):
        """
        Pass a SINGLE hand's data list and the landmark index (0-20).
        Example: landmark_index=8 for INDEX_FINGER_TIP.
        """
        if single_hand_data and 0 <= landmark_index < len(single_hand_data):
            return single_hand_data[landmark_index]
        return None

    def close(self):
        self.model.close()