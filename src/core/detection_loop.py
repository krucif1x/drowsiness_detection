import logging
import cv2
from src.calibration.main_logic import EARCalibrator
from src.utils.ui.visualization import Visualizer
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage
from src.utils.calibration.calculation import EAR, MAR
from src.utils.constants import L_EAR, R_EAR, M_MAR
from src.mediapipe.head_pose import HeadPoseEstimator 
from src.mediapipe.hand import MediaPipeHandsWrapper
from src.detectors.drowsiness import DrowsinessDetector
from src.detectors.distraction import DistractionDetector
from src.detectors.fainting import FaintingDetector  # NEW: integrate fainting
from src.detectors.expression import MouthExpressionClassifier 

log = logging.getLogger(__name__)

class DetectionLoop:
    def __init__(self, camera, face_mesh, buzzer, user_manager, system_logger, vehicle_vin, fps, detector_config_path, initial_user_profile=None, **kwargs):
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.logger = system_logger
        self.visualizer = Visualizer()
        self._post_calibration_cooldown = 0  # Add this line
        
        # Initialize Core Calibrator
        self.ear_calibrator = EARCalibrator(self.camera, self.face_mesh, self.user_manager)
        
        # Initialize Detectors
        self.detector = DrowsinessDetector(self.logger, fps, detector_config_path)
        
        # Initialize Distraction Detector
        self.distraction_detector = DistractionDetector(
            fps=fps,
            camera_pitch=0.0,  
            camera_yaw=0.0,
            config_path=detector_config_path
        )
        
        # NEW: Fainting detector
        self.fainting_detector = FaintingDetector(
            fps=fps,
            camera_pitch=0.0,
            config_path=detector_config_path
        )
        
        # Link detectors for Fainting Logic (legacy note kept; fainting is now separate)
        self.detector.set_last_frame(None)  # ensure safe default
        
        self.head_pose_estimator = HeadPoseEstimator()
        self.expression_classifier = MouthExpressionClassifier()
        
        # Hand Wrapper (Max 2 hands)
        self.hand_wrapper = MediaPipeHandsWrapper(max_num_hands=2)
        
        # Calculators
        self.ear_calculator = EAR() 
        self.mar_calculator = MAR()
        self.fps_tracker = FpsTracker()
        self.ear_smoother = RollingAverage(1.0, fps)
        
        self.user = initial_user_profile
        self.current_mode = 'DETECTING' if initial_user_profile else 'WAITING_FOR_USER'
        
        if self.user:
            self.detector.set_active_user(self.user)
            
        self._frame_idx = 0
        self._show_debug_deltas = False 
        
        # Buffer to prevent instant calibration
        self.recognition_patience = 0
        self.RECOGNITION_THRESHOLD = 45  # ~1.5 seconds

        # --- OPTIMIZATION VARIABLES ---
        self.HAND_INFERENCE_INTERVAL = 5     # Run hand detection every 5 frames
        self.USER_SEARCH_INTERVAL = 30       # Search for user every 30 frames (1 sec)
        self._cached_hands_data = []         # Store hand data for skipped frames

    def run(self):
        log.info("Starting Optimized Detection Loop...")
        try:
            while True:
                self.process_frame()
                
                # --- CRITICAL FIX: GUI EVENT LOOP ---
                # cv2.waitKey(1) processes window events. Without this, the window freezes.
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # Q or ESC to quit
                    break
                elif key == ord('d'):
                    self._show_debug_deltas = not self._show_debug_deltas
                # ------------------------------------

        finally:
            self.hand_wrapper.close()
            cv2.destroyAllWindows()
            self.camera.release()
            
    def process_frame(self):
        frame = self.camera.read()  # Already RGB from camera.py
        if frame is None: 
            return

        self._frame_idx += 1
        fps = self.fps_tracker.update()

        # Use RGB directly for MediaPipe (no conversion needed)
        results = self.face_mesh.process(frame)

        # Hand inference on RGB (every N frames)
        if self._frame_idx % self.HAND_INFERENCE_INTERVAL == 0:
            self._cached_hands_data = self.hand_wrapper.infer(frame, preprocessed=True)
        hands_data = self._cached_hands_data

        # Convert to BGR for OpenCV display
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.current_mode == 'WAITING_FOR_USER':
            self.face_recognition(frame, display, results)
        elif self.current_mode == 'DETECTING':
            self.detection(frame, display, results, hands_data, fps)

        # Draw mode indicator
        cv2.putText(display, f"MODE: {self.current_mode}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Display BGR frame directly (no conversion needed)
        cv2.imshow("Drowsiness System", display)


    def detection(self, frame, display, results, hands_data, fps):
        if not results.multi_face_landmarks: 
            self.visualizer.draw_no_face_text(display)
            return
        
        h, w = frame.shape[:2]
        raw_lms = results.multi_face_landmarks[0]
        
        # Head Pose (degrees)
        pose = self.head_pose_estimator.calculate_pose(raw_lms, w, h)
        pitch, yaw, roll = pose if pose else (0.0, 0.0, 0.0)
        
        # Face Landmarks (Pixels)
        lms = [(int(l.x*w), int(l.y*h)) for l in raw_lms.landmark]
        coords = {
            'left_eye': [lms[i] for i in L_EAR], 
            'right_eye': [lms[i] for i in R_EAR], 
            'mouth': [lms[i] for i in M_MAR]
        }
        
        # Face Center (Normalized)
        nose_tip = raw_lms.landmark[1]
        face_center_norm = (nose_tip.x, nose_tip.y)
        
        # Metrics
        left = self.ear_calibrator.ear_calculator.calculate(coords['left_eye'])
        right = self.ear_calibrator.ear_calculator.calculate(coords['right_eye'])
        mar = self.mar_calculator.calculate(coords['mouth'])
        ear = (left + right) / 2.0
        avg_ear = self.ear_smoother.update(ear)
        
        # Expression
        expr = self.expression_classifier.classify(lms, h, hands_data=hands_data)

        # Drowsiness
        self.detector.set_last_frame(frame)
        drowsy_status, drowsy_color = self.detector.detect(
            avg_ear, mar, expr, 
            hands_data=hands_data, 
            face_center=face_center_norm,
            pitch=pitch 
        )
        drowsy_state = self.detector.get_detailed_state()
        is_drowsy = drowsy_state.get('is_drowsy', False)
        eyes_closed = self.detector.states.get('EYES_CLOSED', False)

        # Normalize hands to 0..1 for distraction detector
        norm_hands = []
        if hands_data:
            for hand in hands_data:
                norm_hand = [(pt[0] / w, pt[1] / h) for pt in hand]
                norm_hands.append(norm_hand)

        # Distraction with normalized hands
        is_distracted, should_log_distraction = self.distraction_detector.analyze(
            pitch, yaw, roll,
            hands=norm_hands,  # ✅ Normalized
            face=face_center_norm
        )

        # Fainting - simplified (no phone/wheel context in new detector)
        self.fainting_detector.set_context(
            drowsy=is_drowsy,
            eyes_closed=eyes_closed,
            phone=False,  # ✅ Removed context dependency
            wheel_count=0  # ✅ Removed context dependency
        )
        is_fainting, faint_is_new = self.fainting_detector.analyze(
            pitch=pitch, yaw=yaw, roll=roll, hands=norm_hands, face_center=face_center_norm
        )

        # Status & UI Logic
        final_status = drowsy_status
        final_color = drowsy_color

        if "CALIBRATING" in drowsy_status:
            final_status = drowsy_status 
            final_color = (255, 255, 0)
            if is_distracted:
                cv2.putText(display, "LOOK FORWARD", (w//2 - 100, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        elif is_fainting:
            final_status = "FAINTING!"
            final_color = (255, 0, 255)
            if faint_is_new:
                self.logger.alert("fainting")
                self.logger.log_event(
                    getattr(self.user, 'user_id', 0),
                    "FAINTING",
                    0.0,
                    float(self.fainting_detector.faint_probability),
                    frame
                )
        elif "SLEEP" in drowsy_status or "YAWN" in drowsy_status or "DROWSY" in drowsy_status:
            pass
        elif is_distracted:
            reason = self.distraction_detector.distraction_type
            
            if "BOTH HANDS" in reason:
                final_status = "BOTH HANDS VISIBLE!"
                final_color = (0, 0, 255)  # Red
            elif "ONE HAND" in reason:
                final_status = "ONE HAND VISIBLE"
                final_color = (0, 165, 255)  # Orange
            elif "ASIDE" in reason:
                final_status = "LOOKING ASIDE"
                final_color = (0, 255, 255)  # Yellow
            elif "DOWN" in reason:
                final_status = "LOOKING DOWN"
                final_color = (0, 255, 255)  # Yellow
            elif "UP" in reason:
                final_status = "LOOKING UP"
                final_color = (0, 255, 255)  # Yellow
            else:
                final_status = "DISTRACTED"
                final_color = (0, 0, 255)
            
            if should_log_distraction:
                log.warning(f"Logging distraction: {reason}")
                self.logger.alert("distraction")
                self.logger.log_event(self.user.user_id, f"DISTRACTION_{reason}", 2.5, 0.0, frame)
        
        
        user_label = f"User {self.user.user_id}" if self.user else "User ?"
        self.visualizer.draw_detection_hud(display, user_label, final_status, final_color, fps, avg_ear, mar, 0, expr, (pitch, yaw, roll))


    def face_recognition(self, frame, display, results):
        if hasattr(self, "_post_calibration_cooldown") and self._post_calibration_cooldown > 0:
            self._post_calibration_cooldown -= 1
            
        if not results.multi_face_landmarks:
            self.visualizer.draw_no_user_text(display)
            self.recognition_patience = 0
            return

        user = self.user_manager.find_best_match(frame)
        if user:
            log.info(f"User identified: {user.user_id}")
            self.user = user
            self.detector.set_active_user(user)
            self.current_mode = 'DETECTING'
            self.recognition_patience = 0
            return

        self.recognition_patience += 1
        self.visualizer.draw_no_user_text(display)

        if self._post_calibration_cooldown > 0:
            return

        if self.recognition_patience >= self.RECOGNITION_THRESHOLD:
            self.calibration(frame)
            return
            
    def calibration(self, frame):
        log.info("Starting Calibration...")
        self.logger.stop_alert()

        result_threshold = self.ear_calibrator.calibrate()
        
        
        # Ensure any calibration windows are closed
        try:
            cv2.destroyWindow("Calibration")
        except:
            pass
        
        if result_threshold is not None and isinstance(result_threshold, float):
            log.info(f"Calibration Success. Threshold: {result_threshold:.3f}")

            new_id = len(self.user_manager.users) + 1
            fresh_frame = self.camera.read()
            if fresh_frame is None:
                fresh_frame = frame

            new_user = self.user_manager.register_new_user(fresh_frame, result_threshold, new_id)

            if new_user:
                log.info(f"New User Registered: ID {new_user.user_id}")
                self.user = new_user
                self.detector.set_active_user(new_user)
                self.current_mode = 'DETECTING'
                log.info("Switched to DETECTING mode after registration.")
            else:
                log.error("Failed to register new user.")
                self.current_mode = 'WAITING_FOR_USER'
        else:
            log.warning("Calibration failed.")
            self.current_mode = 'WAITING_FOR_USER'

        # Reset patience and add a cooldown to avoid immediate recalibration
        self.recognition_patience = 0
        self._post_calibration_cooldown = getattr(self, "_post_calibration_cooldown", 0)
        self._post_calibration_cooldown = 45  # ~1.5s at 30 FPS