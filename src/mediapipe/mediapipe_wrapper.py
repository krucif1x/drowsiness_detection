import cv2
import mediapipe as mp
import numpy as np

class MediaPipeFaceModel:
    """
    A wrapper for the MediaPipe Face Mesh solution.
    Handles initialization, preprocessing, and inference.
    """
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        # Initialize the MediaPipe Face Mesh solution
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, image: np.ndarray):
        """
        Run the model on an image.
        
        Returns:
            The raw MediaPipe results object (containing multi_face_landmarks).
        """
        
        # 2. Run inference
        results = self.face_mesh.process(image)
        
        return results

    def close(self):
        """Release resources."""
        self.face_mesh.close()