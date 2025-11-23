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
        """
        Build normalized encoding matrix from user profiles.
        
        This matrix is used for efficient batch distance calculations.
        All encodings are normalized to unit vectors for consistent distances.
        
        Args:
            users: List of UserProfile objects with face encodings
        """
        if not users:
            self._mat = None
            self._user_ids = []
            logging.debug("Empty user list - matrix cleared")
            return
        
        try:
            # Stack all encodings into matrix (N_users x 512)
            encodings = [u.face_encoding for u in users]
            self._mat = np.stack(encodings, axis=0).astype(np.float32)
            
            # CRITICAL: Normalize each encoding to unit vector
            # This ensures distance calculations are consistent
            # Formula: normalized_vec = vec / ||vec||
            norms = np.linalg.norm(self._mat, axis=1, keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
            self._mat = self._mat / norms
            
            # Store user IDs for debugging
            self._user_ids = [u.user_id for u in users]
            
            logging.info(
                f"Matrix built: {self._mat.shape[0]} users, "
                f"{self._mat.shape[1]} dimensions, "
                f"avg_norm={np.mean(np.linalg.norm(self._mat, axis=1)):.4f}"
            )
            
        except Exception as e:
            logging.error(f"Failed to build matrix: {e}")
            self._mat = None
            self._user_ids = []

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

    def find_top_k_matches(
        self, 
        encoding: np.ndarray, 
        users: List[UserProfile], 
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[UserProfile, float]]:
        """
        Find top-k closest matches for given encoding.
        Useful for debugging and seeing similar users.
        
        Args:
            encoding: Query face encoding
            users: List of user profiles
            k: Number of top matches to return
            threshold: Optional distance threshold to filter results
            
        Returns:
            List of (user, distance) tuples, sorted by distance (ascending)
        """
        if not users or self._mat is None or encoding is None:
            return []
        
        # Normalize query
        query = encoding / (np.linalg.norm(encoding) + 1e-8)
        
        # Calculate distances
        distances = self._euclidean_distance(query)
        
        # Get top-k indices
        k = min(k, len(users))
        top_k_indices = np.argpartition(distances, k-1)[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
        
        # Build results
        results = []
        for idx in top_k_indices:
            distance = float(distances[idx])
            if threshold is None or distance <= threshold:
                results.append((users[idx], distance))
        
        return results

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

    def get_distance_matrix(self, users: List[UserProfile]) -> np.ndarray:
        """
        Calculate pairwise distances between all users.
        Useful for analyzing user separability and detecting duplicates.
        
        Returns:
            NxN matrix where element [i,j] is distance between user i and user j
        """
        if not users or self._mat is None:
            return np.array([])
        
        n = self._mat.shape[0]
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(self._mat[i] - self._mat[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix

    def analyze_separability(self, users: List[UserProfile]) -> dict:
        """
        Analyze how well users are separated in embedding space.
        Helps identify potential duplicate registrations or threshold issues.
        
        Returns:
            Dictionary with analysis results:
            - min_inter_user_distance: Smallest distance between different users
            - mean_inter_user_distance: Average distance between users
            - closest_pair: (user_id_1, user_id_2, distance) of most similar users
            - recommendation: Suggested threshold based on data
        """
        if not users or len(users) < 2:
            return {'error': 'Need at least 2 users for analysis'}
        
        dist_matrix = self.get_distance_matrix(users)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        
        if len(upper_triangle) == 0:
            return {'error': 'Unable to calculate distances'}
        
        min_dist = float(np.min(upper_triangle))
        mean_dist = float(np.mean(upper_triangle))
        std_dist = float(np.std(upper_triangle))
        
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dist_matrix + np.eye(len(users)) * 999), dist_matrix.shape)
        closest_pair = (
            self._user_ids[min_idx[0]] if self._user_ids else min_idx[0],
            self._user_ids[min_idx[1]] if self._user_ids else min_idx[1],
            min_dist
        )
        
        # Suggest threshold (conservative: halfway between 0 and min distance)
        suggested_threshold = min_dist * 0.7
        
        analysis = {
            'num_users': len(users),
            'min_inter_user_distance': min_dist,
            'mean_inter_user_distance': mean_dist,
            'std_inter_user_distance': std_dist,
            'closest_pair': closest_pair,
            'suggested_threshold': suggested_threshold,
            'recommendation': self._get_threshold_recommendation(min_dist, mean_dist)
        }
        
        logging.info(
            f"Separability Analysis: {analysis['num_users']} users, "
            f"min_dist={min_dist:.4f}, mean_dist={mean_dist:.4f}, "
            f"suggested_threshold={suggested_threshold:.4f}"
        )
        
        return analysis

    def _get_threshold_recommendation(self, min_dist: float, mean_dist: float) -> str:
        """Generate threshold recommendation based on distance statistics."""
        if min_dist < 0.3:
            return (
                f"⚠️  WARNING: Very similar users detected (min_dist={min_dist:.3f}). "
                "Consider re-registering users or checking for duplicates. "
                f"Recommended threshold: {min_dist * 0.7:.3f}"
            )
        elif min_dist < 0.5:
            return (
                f"✓ Good separation. Recommended threshold: {min_dist * 0.7:.3f} to {min_dist * 0.85:.3f}"
            )
        else:
            return (
                f"✓ Excellent separation. Can use higher threshold for convenience: "
                f"{min_dist * 0.5:.3f} to {min_dist * 0.7:.3f}"
            )

    def get_statistics(self) -> dict:
        """Get matching statistics for monitoring."""
        stats = self._stats.copy()
        
        if stats['avg_match_distance']:
            stats['avg_match_distance'] = np.mean(stats['avg_match_distance'])
        else:
            stats['avg_match_distance'] = 0.0
        
        if stats['avg_reject_distance']:
            stats['avg_reject_distance'] = np.mean(stats['avg_reject_distance'])
        else:
            stats['avg_reject_distance'] = 0.0
        
        if stats['total_comparisons'] > 0:
            stats['match_rate'] = stats['successful_matches'] / stats['total_comparisons']
        else:
            stats['match_rate'] = 0.0
        
        return stats

    def reset_statistics(self):
        """Reset statistics counters."""
        self._stats = {
            'total_comparisons': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_match_distance': [],
            'avg_reject_distance': []
        }
        logging.info("Matcher statistics reset")