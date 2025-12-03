import numpy as np
import cv2
import time
from typing import Optional, Tuple, Union
from queue import Queue

from src.utils.calibration.calculation import EAR
from src.services.user_manager import UserManager
from src.utils.calibration.thread import start_user_check_thread, stop_user_check_thread
from src.utils.calibration.processor import landmark_processor
from src.utils.calibration.average_ear import average_ear
from src.utils.calibration.feedback import feedback

class EARCalibrator:
    """
    Optimized EAR calibration with efficient processing and robust filtering.
    """
    CALIBRATION_DURATION_S = 10
    FACE_LOST_TIMEOUT_S = 3.5
    MIN_VALID_SAMPLES = 20

    USER_CHECK_INTERVAL = 15
    DISPLAY_UPDATE_INTERVAL = 3
    EAR_BOUNDS = (0.06, 0.60)
    STABILITY_WINDOW = 20
    PREALLOCATE_SIZE = 300

    def __init__(self, camera, face_mesh, user_manager: UserManager):
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.ear_calculator = EAR()
        self._ear_buffer = np.zeros(self.PREALLOCATE_SIZE, dtype=np.float32)
        self._ear_count = 0
        self._user_check_queue = Queue(maxsize=1)
        self._user_check_result = None
        self._user_check_thread = None

    def calibrate(self) -> Union[None, float, Tuple[str, object]]:
        """
        Perform EAR calibration with optimized processing.
        Returns:
            - float: Calibrated EAR threshold
            - ('user_swap', UserProfile): If existing user detected
            - None: If calibration failed/cancelled
        """
        print(f"\n--- EAR Calibration: Look at the camera for {self.CALIBRATION_DURATION_S} seconds. ---")
        time.sleep(1.0)

        self._ear_count = 0
        start_time = time.time()
        face_lost_start_time = None
        frame_count = 0

        calibration_duration = self.CALIBRATION_DURATION_S
        face_lost_timeout = self.FACE_LOST_TIMEOUT_S
        user_check_interval = self.USER_CHECK_INTERVAL
        display_update_interval = self.DISPLAY_UPDATE_INTERVAL

        self.manage_user_check_thread(start=True)

        try:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= calibration_duration:
                    break

                frame = self.camera.read()
                if frame is None:
                    continue

                frame_count += 1
                feedback_frame = cv2.flip(frame, 1)

                if frame_count % user_check_interval == 0:
                    if self._user_check_queue.empty():
                        self._user_check_queue.put(feedback_frame.copy())

                if self._user_check_result is not None:
                    print(f"Recognized user '{getattr(self._user_check_result, 'full_name', self._user_check_result.user_id)}' detected.")
                    return ('user_swap', self._user_check_result)

                rgb_frame = feedback_frame
                results = self.face_mesh.process(rgb_frame)
                ear, status_msg = self._process_landmarks_optimized(results, rgb_frame.shape)

                if status_msg == "Face not detected":
                    if face_lost_start_time is None:
                        face_lost_start_time = time.time()
                    elif time.time() - face_lost_start_time > face_lost_timeout:
                        print("Calibration failed: Face was not detected for too long.")
                        return None
                else:
                    face_lost_start_time = None

                if frame_count % display_update_interval == 0:
                    display_frame_bgr = cv2.cvtColor(feedback_frame, cv2.COLOR_RGB2BGR)
                    self.feedback(display_frame_bgr, ear, elapsed_time, status_msg, self._ear_count)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Calibration cancelled by user.")
                    return None

        finally:
            self.manage_user_check_thread(start=False)

        return self.average_ear()

    def manage_user_check_thread(self, start=True):
        if start:
            start_user_check_thread(self)
        else:
            stop_user_check_thread(self)

    def _process_landmarks_optimized(self, results, frame_shape):
        return landmark_processor(self, results, frame_shape)

    def average_ear(self) -> Optional[float]:
        return average_ear(self)

    def feedback(self, frame, ear, elapsed, status_msg, num_samples):
        feedback(self, frame, ear, elapsed, status_msg, num_samples)