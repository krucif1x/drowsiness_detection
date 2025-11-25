import numpy as np
import torch
import torch.nn.functional as F
import logging
import time
from typing import Optional, Tuple
from src.utils.face_recognition.image_validator import ImageValidator
from src.utils.face_recognition.model_loader import FaceModelLoader

class FaceEncodingExtractor:
    """
    Extracts normalized 512-dimensional face embeddings using FaceNet (MTCNN + InceptionResNet).
    
    Process:
    1. Detect faces using MTCNN (Multi-task Cascaded Convolutional Networks)
    2. Select highest confidence face
    3. Extract 512-dim embedding using InceptionResNet
    4. Normalize to unit vector for consistent distance calculations
    """
    
    def __init__(
        self, 
        model_loader: FaceModelLoader, 
        validator: ImageValidator, 
        device: torch.device,
        min_detection_prob: float = 0.95,  # Increased from 0.90 for better quality
        extract_all_faces: bool = False      # Option to extract multiple faces
    ):
        """
        Initialize face encoding extractor.
        
        Args:
            model_loader: Loaded FaceNet models (MTCNN + ResNet)
            validator: Image preprocessing validator
            device: torch device (cuda/cpu)
            min_detection_prob: Minimum face detection confidence (0-1)
            extract_all_faces: If True, extract all detected faces instead of best only
        """
        self.models = model_loader
        self.validator = validator
        self.device = device
        self.min_detection_prob = min_detection_prob
        self.extract_all_faces = extract_all_faces
        
        # Statistics for monitoring
        self._stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_no_face': 0,
            'failed_low_confidence': 0,
            'failed_preprocessing': 0,
            'avg_extraction_time_ms': []
        }
        
        logging.info(
            f"FaceEncodingExtractor initialized: "
            f"min_prob={min_detection_prob:.2f}, "
            f"device={device}"
        )

    def extract(self, frame, return_metadata: bool = False) -> Optional[np.ndarray]:
        """
        Extract face encoding from frame.

        Args:
            frame: Input image (numpy array from camera or PIL Image)
            return_metadata: If True, return (encoding, metadata) tuple

        Returns:
            512-dimensional normalized face encoding, or None if no valid face found
            If return_metadata=True: (encoding, metadata_dict) or (None, metadata_dict)
        """
        import numpy as np
        from PIL import Image

        t0 = time.time()
        metadata = {
            'detection_prob': 0.0,
            'num_faces_detected': 0,
            'extraction_time_ms': 0.0,
            'face_selected_index': -1
        }

        self._stats['total_extractions'] += 1

        # Step 1: Preprocess and validate frame
        try:
            frame = self.validator.preprocess(frame)
        except ValueError as e:
            logging.debug(f"Preprocessing failed: {e}")
            self._stats['failed_preprocessing'] += 1
            return (None, metadata) if return_metadata else None

        # Step 2: Convert numpy array to PIL Image if needed
        if isinstance(frame, np.ndarray):
            pil_frame = Image.fromarray(frame)
        else:
            pil_frame = frame  # Already PIL Image

        # Step 3: Detect faces using MTCNN
        try:
            faces, probs = self.models.mtcnn(pil_frame, return_prob=True)
        except TypeError:
            res = self.models.mtcnn(pil_frame)
            faces, probs = (res, None) if res is not None else (None, None)

        # Step 4: Validate face detection results
        if faces is None or not isinstance(faces, torch.Tensor):
            logging.debug("No faces detected in frame")
            self._stats['failed_no_face'] += 1
            return (None, metadata) if return_metadata else None

        if faces.ndim == 3:
            faces = faces.unsqueeze(0)

        if faces.shape[0] == 0:
            logging.debug("Empty faces tensor")
            self._stats['failed_no_face'] += 1
            return (None, metadata) if return_metadata else None

        metadata['num_faces_detected'] = faces.shape[0]

        # Step 5: Select best face based on detection probability
        idx = 0
        detection_prob = 1.0

        if probs is not None and len(probs) == faces.shape[0]:
            probs_array = np.array(probs, dtype=np.float32).reshape(-1)
            idx = int(np.argmax(probs_array))
            detection_prob = float(probs_array[idx])

            metadata['detection_prob'] = detection_prob
            metadata['face_selected_index'] = idx

            if detection_prob < self.min_detection_prob:
                logging.debug(
                    f"Face detection confidence too low: {detection_prob:.3f} "
                    f"< {self.min_detection_prob:.3f}"
                )
                self._stats['failed_low_confidence'] += 1
                return (None, metadata) if return_metadata else None

        # Step 6: Extract embedding using InceptionResNet
        face = faces[idx].unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.models.resnet(face)
            embedding = F.normalize(embedding, p=2, dim=1)

        encoding = embedding.cpu().numpy().flatten()

        # Step 7: Final validation
        if not self.validate_encoding(encoding):
            logging.warning("Extracted encoding failed validation")
            return (None, metadata) if return_metadata else None

        extraction_time = (time.time() - t0) * 1000
        metadata['extraction_time_ms'] = extraction_time
        self._stats['successful_extractions'] += 1
        self._stats['avg_extraction_time_ms'].append(extraction_time)

        logging.debug(
            f"✓ Encoding extracted: {extraction_time:.1f}ms, "
            f"prob={detection_prob:.3f}, "
            f"faces={metadata['num_faces_detected']}"
        )

        return (encoding, metadata) if return_metadata else encoding

    def validate_encoding(self, encoding: np.ndarray) -> bool:
        """
        Validate encoding quality and integrity.
        
        Checks:
        - Correct dimensionality (512)
        - No NaN or Inf values
        - Non-zero magnitude
        - Reasonable value range
        """
        if encoding is None:
            return False
        
        # Check dimensions
        if len(encoding) != 512:
            logging.warning(f"Invalid encoding size: {len(encoding)} (expected 512)")
            return False
        
        # Check for invalid values
        if np.isnan(encoding).any():
            logging.warning("Encoding contains NaN values")
            return False
        
        if np.isinf(encoding).any():
            logging.warning("Encoding contains Inf values")
            return False
        
        # Check magnitude (should be ~1.0 after normalization)
        norm = np.linalg.norm(encoding)
        if norm < 0.5 or norm > 1.5:
            logging.warning(f"Encoding norm out of range: {norm:.3f} (expected ~1.0)")
            return False
        
        # Check value range (normalized embeddings should be roughly -1 to 1)
        if np.abs(encoding).max() > 10.0:
            logging.warning(f"Encoding values out of range: max={np.abs(encoding).max():.3f}")
            return False
        
        return True
