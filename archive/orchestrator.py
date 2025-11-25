import logging
import time
import cv2
import numpy as np
from collections import deque
from typing import Optional

from src.utils.ui.visualization import Visualizer
from src.utils.ear.calculation import EAR, MAR
from src.utils.ear.constants import L_EAR, R_EAR, M_MAR
from src.utils.ui.metrics_tracker import update_fps
from src.detectors.drowsiness import DrowsinessDetector
from src.services.user_manager import UserManager
from src.detectors.calibration.main_logic import EARCalibrator
from src.detectors.expression import MouthExpressionClassifier 
from src.infrastructure.data.drowsiness_events.repository import DrowsinessEventRepository

log = logging.getLogger(__name__)

class DetectionLoop:
    """Main detection loop with user management and drowsiness monitoring."""
    
    # Config constants
    BUFFER_DURATION_SECONDS = 1.0
    USER_LOST_THRESHOLD = 30
    UNKNOWN_FACE_CONFIRM_FRAMES = 5
    
    # Identity check
    ID_PROBE_EVERY_N = 60
    SAME_USER_MIN_SIM = 0.70
    SAME_USER_LOW_SIM_STREAK = 5
    MAR_ID_MAX = 0.50

    def __init__(self, camera, face_mesh, buzzer, user_manager: UserManager, initial_user_profile=None, 
                 event_logging_service=None, event_repo=None, vehicle_vin="VIN-0001", fps=30.0, 
                 detector_config_path="config/detector_config.yaml"):
        
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.visualizer = Visualizer()
        
        self.detector = DrowsinessDetector(buzzer, event_logging_service, event_repo, vehicle_vin, fps, detector_config_path)
        self.expression_classifier = MouthExpressionClassifier()
        self.ear_calculator = EAR()
        self.mar_calculator = MAR()
        
        self.fps = fps
        self.prev_time = time.time()
        self.ear_buffer = deque()
        self.ear_buffer_sum = 0.0
        
        self.user = initial_user_profile
        self.current_mode = 'DETECTING' if initial_user_profile else 'WAITING_FOR_USER'
        self.user_lost_counter = 0
        self.unknown_face_counter = 0
        self.is_registering = False
        self._frame_idx = 0
        self._low_sim_streak = 0
        
        self.detector.set_active_user(self.user)
        log.info(f"Starting in {self.current_mode} mode")

    def run(self):
        while True:
            self.process_frame()
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    def process_frame(self):
        rgb_frame = self.camera.read()
        if rgb_frame is None: return

        self.fps, self.prev_time = update_fps(self.prev_time)
        results = self.face_mesh.process(rgb_frame)
        display_frame = rgb_frame.copy()

        if self.current_mode == 'DETECTING':
            self._process_detecting_mode(rgb_frame, display_frame, results)
        elif self.current_mode == 'WAITING_FOR_USER':
            self._process_waiting_mode(rgb_frame, display_frame, results)
        elif self.current_mode == 'NO_USER':
            self._process_no_user_mode(rgb_frame, display_frame, results)

        cv2.imshow("Drowsiness System", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

    def _process_detecting_mode(self, rgb_frame, display_frame, results):
        self._frame_idx += 1

        if not results.multi_face_landmarks:
            self._handle_face_lost(rgb_frame)
            return

        if not self.user: return

        h, w = rgb_frame.shape[:2]
        all_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
        
        coords = self._extract_landmarks(all_landmarks)
        ear, mar = self._calculate_ear_mar(coords)
        if ear is None: return

        mouth_expr = self.expression_classifier.classify(all_landmarks, img_h=h)
        self.detector.set_last_frame(rgb_frame.copy(), color_space="RGB")
        
        self.visualizer.draw_landmarks(display_frame, coords)
        avg_ear = self._update_ear_buffer(ear)
        
        # Identity Verification (Optimized)
        if self._frame_idx % self.ID_PROBE_EVERY_N == 0 and (mar is None or mar < self.MAR_ID_MAX):
            self._verify_user_identity(rgb_frame)
        
        status, color = self.detector.detect(avg_ear, mar, mouth_expr)
        
        self.visualizer.draw_detection_hud(
            display_frame, f"User #{self.user.user_id}", status, color, 
            self.fps, avg_ear, mar, self.detector.counters['BLINK_COUNT'], mouth_expr
        )
        self.user_lost_counter = 0

    def _verify_user_identity(self, rgb_frame):
        """Delegates verification to UserManager."""
        # --- CHANGED: Uses single line method instead of manual math ---
        is_match = self.user_manager.verify_identity(
            rgb_frame, self.user, self.SAME_USER_MIN_SIM
        )
        
        if not is_match:
            self._low_sim_streak += 1
            if self._low_sim_streak >= self.SAME_USER_LOW_SIM_STREAK:
                self._handle_user_swap(rgb_frame)
        else:
            self._low_sim_streak = 0

    def _handle_user_swap(self, rgb_frame):
        best_user = self.user_manager.find_best_match(rgb_frame)
        if best_user and best_user.user_id != self.user.user_id:
            log.info(f"User swap: {self.user.user_id} → {best_user.user_id}")
            self.user = best_user
            self.detector.set_active_user(self.user)
        else:
            log.warning("Unknown face detected. Switching to waiting mode.")
            self.user = None
            self.detector.set_active_user(None)
            self.current_mode = 'WAITING_FOR_USER'
        self._low_sim_streak = 0

    def _handle_face_lost(self, rgb_frame):
        best_user = self.user_manager.find_best_match(rgb_frame)
        if best_user:
            if not self.user or best_user.user_id != self.user.user_id:
                self.user = best_user
                self.detector.set_active_user(self.user)
            self.user_lost_counter = 0
            return

        self.user_lost_counter += 1
        if self.user_lost_counter > self.USER_LOST_THRESHOLD:
            log.warning(f"User face lost for {self.USER_LOST_THRESHOLD} frames.")
            self.user = None
            self.detector.set_active_user(None)
            self.current_mode = "NO_USER"

    def _process_no_user_mode(self, rgb_frame, display_frame, results):
        self._frame_idx += 1
        self.visualizer.draw_no_user_text(display_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = display_frame.shape[:2]
            coords = self._extract_landmarks([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
            self.visualizer.draw_landmarks(display_frame, coords)
            
            if self._frame_idx % 3 == 0:
                best_user = self.user_manager.find_best_match(rgb_frame)
                if best_user:
                    self.user = best_user
                    self.detector.set_active_user(self.user)
                    self.current_mode = "DETECTING"
                    self.user_lost_counter = 0

    def _process_waiting_mode(self, rgb_frame, display_frame, results):
        recognized = self.user_manager.find_best_match(rgb_frame)
        if recognized:
            self.user = recognized
            self.detector.set_active_user(self.user)
            self.current_mode = 'DETECTING'
            self.unknown_face_counter = 0
            self.is_registering = False
            return
        
        if results.multi_face_landmarks:
            if not self.is_registering:
                self.unknown_face_counter += 1
                if self.unknown_face_counter >= self.UNKNOWN_FACE_CONFIRM_FRAMES:
                    self.is_registering = True
                    self.unknown_face_counter = 0
                    self._register_new_user()
        else:
            self.unknown_face_counter = 0

    def _register_new_user(self):
        calibrator = EARCalibrator(self.camera, self.face_mesh, self.user_manager)
        result = calibrator.calibrate()
        
        if isinstance(result, tuple) and result[0] == 'user_swap':
            self.user = result[1]
            self.detector.set_active_user(self.user)
            self.current_mode = 'DETECTING'
            self.is_registering = False
            return
        
        if result is None:
            self.current_mode = 'WAITING_FOR_USER'
            self.is_registering = False
            return
        
        log.info("Calibration successful. Capturing registration sequence...")
        
        # --- CHANGED: Uses User Manager to handle capture loop ---
        new_user = self.user_manager.register_sequence(self.camera, result)
        
        if new_user:
            self.user = new_user
            self.detector.set_active_user(self.user)
            self.current_mode = 'DETECTING'
        else:
            log.error("Registration failed")
            self.current_mode = 'WAITING_FOR_USER'
        
        self.is_registering = False

    def _extract_landmarks(self, all_landmarks: list) -> dict:
        return {
            'left_eye': [all_landmarks[i] for i in L_EAR],
            'right_eye': [all_landmarks[i] for i in R_EAR],
            'mouth': [all_landmarks[i] for i in M_MAR]
        }

    def _calculate_ear_mar(self, coords: dict) -> tuple:
        try:
            left = self.ear_calculator.calculate(coords['left_eye'])
            right = self.ear_calculator.calculate(coords['right_eye'])
            if left is None or right is None: return None, None
            return (left + right) / 2.0, self.mar_calculator.calculate(coords['mouth'])
        except Exception:
            return None, None

    def _update_ear_buffer(self, frame_ear: float) -> float:
        if frame_ear is None: return 0.0
        buffer_size = max(1, int(self.fps * self.BUFFER_DURATION_SECONDS))
        self.ear_buffer.append(frame_ear)
        self.ear_buffer_sum += frame_ear
        while len(self.ear_buffer) > buffer_size:
            self.ear_buffer_sum -= self.ear_buffer.popleft()
        return self.ear_buffer_sum / len(self.ear_buffer) if self.ear_buffer else frame_ear