import numpy as np
import logging
from typing import Optional, Tuple, List
from src.infrastructure.data.models import UserProfile

class SimilarityMatcher:
    """
    Matches face encodings using Euclidean distance (L2 norm).
    
    How it works:
    1. Stores all user encodings in a normalized matrix
    2. Calculates distance between query and all stored encodings
    3. Returns closest match if within threshold
    
    Distance interpretation (for normalized FaceNet embeddings):
    - 0.0-0.4: Very likely same person
    - 0.4-0.6: Likely same person (recommended threshold range)
    - 0.6-0.8: Possibly same person (borderline)
    - 0.8+: Different person
    """
    
    def __init__(self, distance_metric: str = "euclidean"):
        """
        Initialize similarity matcher.
        
        Args:
            distance_metric: Distance calculation method
                - "euclidean": L2 distance (default, recommended for FaceNet)
                - "cosine": Cosine distance (alternative)
        """
        self._mat: Optional[np.ndarray] = None
        self._user_ids: List[int] = []  # Track user IDs for debugging
        self.distance_metric = distance_metric.lower()
        
        # Statistics
        self._stats = {
            'total_comparisons': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_match_distance': [],
            'avg_reject_distance': []
        }
        
        logging.info(f"SimilarityMatcher initialized with {distance_metric} distance")
        
    def build_matrix(self, users: List[UserProfile]):
        """Builds and normalizes the user encoding matrix for matching."""
        if not users:
            self._mat = None
            self._user_ids = []
            return
        encodings = [u.face_encoding for u in users if u.face_encoding is not None]
        self._mat = np.array([e / (np.linalg.norm(e) + 1e-8) for e in encodings])
        self._user_ids = [u.user_id for u in users]

    def best_match(
        self, 
        encoding: np.ndarray, 
        users: List[UserProfile], 
        threshold: float = 0.60,
        return_all_distances: bool = False
    ) -> Tuple[Optional[UserProfile], float]:
        """
        Find best matching user for given encoding.
        
        Args:
            encoding: Query face encoding (512-dim)
            users: List of user profiles to match against
            threshold: Maximum distance for valid match (lower = stricter)
            return_all_distances: If True, return (user, distance, all_distances)
            
        Returns:
            (matched_user, distance) or (None, closest_distance) if no match
            
        Note: LOWER distance = BETTER match
        """
        self._stats['total_comparisons'] += 1
        
        # Validation
        if not users or self._mat is None:
            logging.debug("No users in database for matching")
            return (None, 99.9, None) if return_all_distances else (None, 99.9)
        
        if encoding is None or len(encoding) != 512:
            logging.warning(f"Invalid encoding for matching: {encoding.shape if encoding is not None else 'None'}")
            return (None, 99.9, None) if return_all_distances else (None, 99.9)
        
        # Check for invalid values
        if np.isnan(encoding).any() or np.isinf(encoding).any():
            logging.warning("Encoding contains NaN or Inf values")
            return (None, 99.9, None) if return_all_distances else (None, 99.9)
        
        # Normalize query encoding to unit vector
        query = encoding / (np.linalg.norm(encoding) + 1e-8)
        
        # Calculate distances using selected metric
        if self.distance_metric == "cosine":
            distances = self._cosine_distance(query)
        else:  # euclidean (default)
            distances = self._euclidean_distance(query)
        
        # Find closest match
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        
        # Log distance distribution for debugging
        if len(distances) > 1:
            logging.debug(
                f"Distance stats - Best: {best_distance:.4f}, "
                f"Mean: {np.mean(distances):.4f}, "
                f"Std: {np.std(distances):.4f}, "
                f"2nd Best: {np.partition(distances, 1)[1]:.4f}"
            )
        
        # Check against threshold
        if best_distance <= threshold:
            self._stats['successful_matches'] += 1
            self._stats['avg_match_distance'].append(best_distance)
            
            matched_user = users[best_idx]
            logging.debug(
                f"✓ Match found: User {matched_user.user_id} "
                f"(distance={best_distance:.4f}, threshold={threshold:.4f})"
            )
            
            if return_all_distances:
                return matched_user, best_distance, distances
            return matched_user, best_distance
        else:
            self._stats['failed_matches'] += 1
            self._stats['avg_reject_distance'].append(best_distance)
            
            logging.debug(
                f"✗ No match: Closest distance {best_distance:.4f} > threshold {threshold:.4f}"
            )
            
            if return_all_distances:
                return None, best_distance, distances
            return None, best_distance

    def _euclidean_distance(self, query: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean (L2) distance between query and all stored encodings.
        
        Formula: distance = ||query - stored_encoding||
        
        For normalized vectors, this is equivalent to:
        distance = sqrt(2 - 2 * dot(query, stored))
        
        Returns array of distances, one per user.
        """
        # Direct L2 distance calculation
        # Broadcasting: (1, 512) - (N, 512) = (N, 512), then norm across axis=1
        distances = np.linalg.norm(self._mat - query, axis=1)
        return distances

    def _cosine_distance(self, query: np.ndarray) -> np.ndarray:
        """
        Calculate cosine distance between query and all stored encodings.
        
        Formula: distance = 1 - cosine_similarity
        where cosine_similarity = dot(query, stored) / (||query|| * ||stored||)
        
        For unit vectors: cosine_similarity = dot(query, stored)
        
        Returns array of distances, one per user.
        """
        # Cosine similarity for normalized vectors
        similarities = np.dot(self._mat, query)
        
        # Convert to distance (0 = identical, 2 = opposite)
        distances = 1.0 - similarities
        return distances
