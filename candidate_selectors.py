"""
Candidate selection strategies for scene segmentation.

This module provides different strategies for selecting candidate transition points
from sentence embeddings. All selectors implement the CandidateSelector interface.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, savgol_filter


class CandidateSelector(ABC):
    """Abstract base class for candidate selection strategies."""

    @abstractmethod
    def select_candidates(self, embeddings: np.ndarray) -> List[int]:
        """
        Select candidate transition points from embeddings.

        Args:
            embeddings: Array of shape (num_sentences, embedding_dim)

        Returns:
            List of candidate indices (positions between sentences)
        """
        pass


class GaussianCandidateSelector(CandidateSelector):
    """
    Gaussian smoothing-based candidate selection (original approach).

    Computes delta curve from embeddings, applies Gaussian smoothing,
    then finds local minima.
    """

    def __init__(self, sigma: float = 0.8, diff_metric: str = "cosine",
                 target: str = "minima"):
        """
        Initialize Gaussian candidate selector.

        Args:
            sigma: Standard deviation for Gaussian kernel (smoothing strength)
            diff_metric: Distance metric - "2norm", "cosine", etc.
            target: What to detect - "minima", "maxima", or "both"
        """
        self.sigma = sigma
        self.diff_metric = diff_metric
        self.target = target

    def select_candidates(self, embeddings: np.ndarray) -> List[int]:
        """Select candidates using Gaussian smoothing + local minima detection."""
        # Compute delta curve
        deltaY = self._compute_delta_curve(embeddings)

        # Apply Gaussian smoothing
        deltaY_smoothed = gaussian_filter1d(deltaY, self.sigma)

        # Find local minima/maxima
        minima_indices = argrelextrema(deltaY_smoothed, np.less)[0]
        maxima_indices = argrelextrema(deltaY_smoothed, np.greater)[0]

        if self.target == "minima":
            return minima_indices.tolist()
        elif self.target == "maxima":
            return maxima_indices.tolist()
        elif self.target == "both":
            return np.concatenate((minima_indices, maxima_indices)).tolist()
        else:
            return minima_indices.tolist()

    def _compute_delta_curve(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute difference curve between consecutive embeddings.

        Args:
            embeddings: Array of sentence embeddings

        Returns:
            Array of differences between consecutive embeddings
        """
        deltaY = []

        # Parse metric
        import re
        if bool(re.fullmatch(r"(inf|\d+)norm", self.diff_metric)):
            # L-p norm calculation
            if self.diff_metric == "infnorm":
                p = np.inf
            else:
                p = int(self.diff_metric[0])

            for n in range(len(embeddings) - 1):
                deltaY.append(np.linalg.norm(embeddings[n] - embeddings[n+1], p))

        elif "cos" in self.diff_metric:
            # Cosine similarity calculation
            for n in range(len(embeddings) - 1):
                norm_product = (np.linalg.norm(embeddings[n]) *
                               np.linalg.norm(embeddings[n+1]))
                if norm_product != 0:
                    deltaY.append(np.dot(embeddings[n], embeddings[n+1]) / norm_product)
                else:
                    deltaY.append(0)

        return np.array(deltaY)


class SavitzkyGolayCandidateSelector(CandidateSelector):
    """
    Savitzky-Golay polynomial smoothing-based candidate selection.

    Applies polynomial smoothing to the delta curve, then finds local minima.
    This approach preserves peak shapes better than Gaussian smoothing.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3,
                 distance_metric: str = "cosine"):
        """
        Initialize Savitzky-Golay candidate selector.

        Args:
            window_length: Length of filter window (must be odd, >= polyorder+2)
            polyorder: Order of polynomial for fitting
            distance_metric: Distance metric for computing delta curve
        """
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if window_length <= polyorder:
            raise ValueError("window_length must be greater than polyorder")

        self.window_length = window_length
        self.polyorder = polyorder
        self.distance_metric = distance_metric

    def select_candidates(self, embeddings: np.ndarray) -> List[int]:
        """Select candidates using Savitzky-Golay smoothing + local minima detection."""
        # Compute delta curve
        deltaY = self._compute_delta_curve(embeddings)

        # Handle edge case: not enough points for window
        if len(deltaY) < self.window_length:
            # Fall back to smaller window or no smoothing
            window_length = len(deltaY) if len(deltaY) % 2 == 1 else len(deltaY) - 1
            if window_length > self.polyorder:
                deltaY_smoothed = savgol_filter(deltaY, window_length, self.polyorder)
            else:
                deltaY_smoothed = deltaY
        else:
            # Apply Savitzky-Golay smoothing
            deltaY_smoothed = savgol_filter(deltaY, self.window_length, self.polyorder)

        # Find local minima
        minima_indices = argrelextrema(deltaY_smoothed, np.less)[0]

        return minima_indices.tolist()

    def _compute_delta_curve(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute difference curve between consecutive embeddings.

        Args:
            embeddings: Array of sentence embeddings

        Returns:
            Array of differences between consecutive embeddings
        """
        deltaY = []

        if "cos" in self.distance_metric:
            # Cosine similarity calculation
            for n in range(len(embeddings) - 1):
                norm_product = (np.linalg.norm(embeddings[n]) *
                               np.linalg.norm(embeddings[n+1]))
                if norm_product != 0:
                    deltaY.append(np.dot(embeddings[n], embeddings[n+1]) / norm_product)
                else:
                    deltaY.append(0)
        else:
            # Default to L2 norm
            for n in range(len(embeddings) - 1):
                deltaY.append(np.linalg.norm(embeddings[n] - embeddings[n+1]))

        return np.array(deltaY)
