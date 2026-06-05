"""
tgpsf.py  -  Target-Guided Projection and Similarity Filtering (Stage 1)
=========================================================================
Algorithm 1 of LearnSense-LearnBoost-LearnVerify.

PIPELINE CONTRACT: fit() ONLY on training data.
transform() may be called on any split (train / val / test).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin


class TGPSF(BaseEstimator, TransformerMixin):
    """
    Target-Guided Projection and Similarity Filtering.

    Two passes:
    Pass 1 - Similarity  : keep features with |Pearson(X_j, y)| >= Ts,
                           where Ts = mean(Jsc) + delta * std(Jsc).
    Pass 2 - Projection  : discard surviving features whose projection
                           error onto the target sub-space > epsilon.

    Parameters
    ----------
    delta       : float   Threshold sensitivity.                 Default 0.5
    lambda_reg  : float   Ridge penalty for projection matrix.   Default 0.01
    epsilon     : float   Max projection error (normalised).     Default 0.10
    random_state: int     Seed (for future extensions).         Default 42
    """

    def __init__(self, delta=0.5, lambda_reg=0.01, epsilon=0.10, random_state=42):
        self.delta        = delta
        self.lambda_reg   = lambda_reg
        self.epsilon      = epsilon
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TGPSF":
        """Fit on TRAINING DATA ONLY. Never call on val/test."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_feat = X.shape[1]

        # --- Pass 1: Similarity scoring (Equations 15-16) ----------------
        jsc = np.zeros(n_feat)
        for j in range(n_feat):
            if np.std(X[:, j]) > 1e-9:
                jsc[j] = abs(pearsonr(X[:, j], y)[0])
        self.similarity_scores_ = jsc
        Ts = jsc.mean() + self.delta * jsc.std()
        self.threshold_Ts_ = float(Ts)
        candidates = np.where(jsc >= Ts)[0]

        # --- Pass 2: Projection consistency (Equations 18-21) ------------
        # Closed-form ridge: w_j = (y @ X_j) / (y @ y + lambda)
        yTy    = float(y @ y) + self.lambda_reg
        errors = np.zeros(len(candidates))
        for k, j in enumerate(candidates):
            xj   = X[:, j]
            wj   = float(y @ xj) / yTy
            proj = wj * y
            err  = float(np.mean((xj - proj) ** 2))
            errors[k] = err / (float(np.var(xj)) + 1e-9)   # normalised
        self.projection_errors_ = errors

        kept = candidates[errors <= self.epsilon]
        if len(kept) == 0:                    # fallback: keep best feature
            kept = np.array([int(np.argmax(jsc))])
        self.selected_features_ = kept
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X to the selected feature subset."""
        return np.asarray(X)[:, self.selected_features_]

    def reduction_pct(self) -> float:
        n_orig = len(self.similarity_scores_)
        return 100.0 * (1 - len(self.selected_features_) / n_orig)

    def __repr__(self):
        n = getattr(self, "selected_features_", None)
        sel = f", n_selected={len(n)}" if n is not None else ""
        return (f"TGPSF(delta={self.delta}, lambda_reg={self.lambda_reg}, "
                f"epsilon={self.epsilon}{sel})")
