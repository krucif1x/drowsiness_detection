import logging
import torch
import numpy as np
from typing import List, Optional, Tuple
from threading import Lock
from collections import deque

from src.infrastructure.data.models import UserProfile
from src.infrastructure.data.database import UnifiedDatabase
from src.infrastructure.data.repository import UnifiedRepository
from src.utils.face_recognition.model_loader import FaceModelLoader
from src.utils.face_recognition.image_validator import ImageValidator
from src.utils.face_recognition.encoding_extractor import FaceEncodingExtractor
from src.utils.face_recognition.similarity_matcher import SimilarityMatcher

class UserManager:
    def __init__(
        self, 
        database_file: str = r"data\drowsiness_events.db",
        recognition_threshold: float = 0.5,  # Conservative threshold for better accuracy
        
        # Multi-frame validation reduces false positives significantly
        multi_frame_validation: bool = True,
        min_consistent_frames: int = 10,  # Require 3 consistent detections
        
        # Quality thresholds
        min_face_confidence: float = 0.95,  # Only use high-quality detections
        
        input_color: str = "RGB"
    ):
        logging.info(f"UserManager initializing with database: {database_file}")
        
        self.recognition_threshold = recognition_threshold
        self.input_color = (input_color or "RGB").upper()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Multi-frame validation settings
        self.multi_frame_validation = multi_frame_validation
        self.min_consistent_frames = min_consistent_frames
        self.min_face_confidence = min_face_confidence
        
        # Store recent frame results for validation
        self._recent_matches: deque = deque(maxlen=min_consistent_frames)
        
        self._lock = Lock()
        
        # Components with enhanced configuration
        self.db = UnifiedDatabase(database_file)
        self.repo = UnifiedRepository(self.db)
        self.model_loader = FaceModelLoader(device=str(self.device))
        self.validator = ImageValidator(self.input_color)
        
        # Enhanced encoder with configurable detection threshold
        self.encoder = FaceEncodingExtractor(
            self.model_loader, 
            self.validator, 
            self.device,
            min_detection_prob=min_face_confidence
        )
        
        # Enhanced matcher with distance metric selection
        self.matcher = SimilarityMatcher(distance_metric="euclidean")

        # Caches
        self.users: List[UserProfile] = []
        self._user_id_map: dict[int, UserProfile] = {}
        
        # Statistics for monitoring
        self._match_stats = {
            'total_attempts': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_match_distance': []
        }
        
        self.load_users()
        logging.info(f"UserManager initialized successfully with threshold={recognition_threshold}")

    def load_users(self):
        """Load all user profiles from database into memory cache."""
        try:
            with self._lock:
                self.users = self.repo.load_all_users()
                self._user_id_map = {u.user_id: u for u in self.users}
                self.matcher.build_matrix(self.users)
            logging.info(f"Loaded {len(self.users)} user profile(s)")
        except Exception as e:
            logging.error(f"Error loading users: {e}")
            self.users = []
            self._user_id_map = {}

    def calculate_confidence_score(self, distance: float, threshold: float) -> float:
        """
        Calculate confidence score based on linear interpolation from threshold.
        Returns value between 0-1, where 1 is highest confidence.
        """
        if distance >= threshold:
            return 0.0
        # Linear interpolation: confidence decreases linearly as distance increases
        confidence = 1.0 - (distance / threshold)
        return max(confidence, 0.0)

    def find_best_match(self, image_frame, use_metadata: bool = False) -> Optional[UserProfile]:
        """
        Find best matching user from image frame with enhanced validation.
        Uses multi-frame consistency check to reduce false positives.
        
        Args:
            image_frame: Input frame to match
            use_metadata: If True, log detailed extraction metadata
        """
        if not self.users:
            logging.debug("No users in database")
            return None

        # Extract encoding with optional metadata
        if use_metadata:
            result = self.encoder.extract(image_frame, return_metadata=True)
            if result[0] is None:
                logging.debug(f"Extraction failed: {result[1]}")
                return None
            encoding, metadata = result
            logging.debug(f"Extraction metadata: {metadata}")
        else:
            encoding = self.encoder.extract(image_frame)
            if encoding is None:
                logging.debug("No face detected in frame")
                return None

        with self._lock:
            # Find best match with distance
            best_user, dist = self.matcher.best_match(
                encoding, 
                self.users, 
                threshold=self.recognition_threshold
            )
            
            # Calculate confidence
            confidence = self.calculate_confidence_score(dist, self.recognition_threshold)
            
            # Update statistics
            self._match_stats['total_attempts'] += 1
            if dist < self.recognition_threshold:
                self._match_stats['avg_match_distance'].append(dist)

        # Multi-frame validation for consistency
        if self.multi_frame_validation and best_user:
            self._recent_matches.append((best_user.user_id if best_user else None, dist))
            
            # Check if we have enough frames
            if len(self._recent_matches) < self.min_consistent_frames:
                logging.debug(f"Collecting frames: {len(self._recent_matches)}/{self.min_consistent_frames}")
                return None
            
            # Check consistency across recent frames
            recent_user_ids = [match[0] for match in self._recent_matches]
            most_common_id = max(set(recent_user_ids), key=recent_user_ids.count)
            consistency_count = recent_user_ids.count(most_common_id)
            
            # Require majority consensus
            if consistency_count < (self.min_consistent_frames * 0.6):  # 60% threshold
                logging.debug(f"Inconsistent matches across frames: {recent_user_ids}")
                return None
            
            # If consensus doesn't match current best, return None
            if most_common_id != (best_user.user_id if best_user else None):
                logging.debug(f"Current match inconsistent with recent history")
                return None

        if best_user and dist < self.recognition_threshold:
            logging.info(
                f"✓ MATCH FOUND: User ID {best_user.user_id} "
                f"(Distance: {dist:.4f}, Confidence: {confidence:.2%}, "
                f"Threshold: {self.recognition_threshold})"
            )
            self._match_stats['successful_matches'] += 1
            self.repo.update_last_seen(best_user.user_id)
            return best_user
        else:
            self._match_stats['failed_matches'] += 1
            logging.info(
                f"✗ NO MATCH (Closest: User {best_user.user_id if best_user else 'N/A'}, "
                f"Distance: {dist:.4f}, Confidence: {confidence:.2%}, "
                f"Threshold: {self.recognition_threshold})"
            )
            return None

    def register_new_user(
        self, 
        image_frame, 
        ear_threshold: float, 
        user_id: Optional[int] = None,
        require_multiple_frames: bool = False,
        additional_frames: Optional[List] = None
    ) -> Optional[UserProfile]:
        """
        Register new user with enhanced multi-frame encoding.
        
        Args:
            image_frame: Initial frame for registration
            ear_threshold: EAR threshold for drowsiness detection
            user_id: Optional specific user ID
            require_multiple_frames: Whether to use multiple frames
            additional_frames: List of additional frames to average (if multiple)
        """
        if ear_threshold is None: 
            return None

        # Extract encoding from primary frame
        encoding = self.encoder.extract(image_frame)
        if encoding is None: 
            logging.error("Cannot register: No face encoding extracted from primary frame")
            return None

        # Multi-frame registration for better accuracy
        encodings = [encoding]
        
        if require_multiple_frames and additional_frames:
            logging.info(f"Multi-frame registration with {len(additional_frames)} additional frames...")
            
            for frame in additional_frames:
                enc = self.encoder.extract(frame)
                if enc is not None:
                    encodings.append(enc)
            
            logging.info(f"Collected {len(encodings)} valid encodings from {len(additional_frames) + 1} frames")
        
        # Average encodings for robustness (already normalized by encoder)
        final_encoding = np.mean(encodings, axis=0)
        
        # Normalize the averaged encoding
        final_encoding = final_encoding / (np.linalg.norm(final_encoding) + 1e-8)

        # Check for duplicates with stricter threshold during registration
        registration_threshold = self.recognition_threshold * 0.8  # 20% stricter
        
        with self._lock:
            duplicate, dist = self.matcher.best_match(
                final_encoding, 
                self.users, 
                threshold=registration_threshold
            )

        if duplicate:
            logging.warning(
                f"⚠ DUPLICATE DETECTED: User ID {duplicate.user_id} already exists "
                f"(Distance: {dist:.4f}, Threshold: {registration_threshold:.4f}). "
                f"Returning existing user."
            )
            return duplicate

        # Get next user_id if not provided
        if user_id is None:
            user_id = self.repo.get_next_user_id()

        # Create User Object with validated encoding
        new_user = UserProfile(0, user_id, ear_threshold, final_encoding)

        try:
            # Save to DB
            new_user.id = self.repo.save_user(new_user)
            
            # Update Memory Cache
            with self._lock:
                self.users.append(new_user)
                self._user_id_map[user_id] = new_user
                self.matcher.build_matrix(self.users)
            
            logging.info(
                f"✓ NEW USER REGISTERED: ID={user_id}, "
                f"EAR Threshold={ear_threshold:.2f}, "
                f"Encoding Quality: Valid"
            )
            return new_user
        except Exception as e:
            logging.error(f"✗ User registration failed: {e}")
            return None

    