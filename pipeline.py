"""
pipeline.py  -  Full three-stage LearnSense-LearnBoost-LearnVerify pipeline
============================================================================
Combines TG-PSF (Stage 1), RGEAB-TE (Stage 2), CCSG-PR (Stage 3).
"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .tgpsf   import TGPSF
from .rgeabte import RGEABTE
from .ccsgpr  import CCSGPR


class LearnSenseBoostVerify:
    """
    Full three-stage pipeline.

    Parameters
    ----------
    tgpsf_kwargs   : dict  Passed to TGPSF.
    rgeabte_kwargs : dict  Passed to RGEABTE.
    ccsgpr_kwargs  : dict  Passed to CCSGPR.

    Example
    -------
    >>> model = LearnSenseBoostVerify()
    >>> model.fit(X_train, y_train)
    >>> labels, omega, accepted = model.predict(X_test)
    >>> full_cov_acc = model.score_full_coverage(X_test, y_test)
    """

    def __init__(
        self,
        tgpsf_kwargs:   dict | None = None,
        rgeabte_kwargs: dict | None = None,
        ccsgpr_kwargs:  dict | None = None,
    ):
        self.tgpsf   = TGPSF(  **(tgpsf_kwargs   or {}))
        self.rgeabte = RGEABTE(**(rgeabte_kwargs  or {}))
        self.ccsgpr  = CCSGPR( **(ccsgpr_kwargs   or {}))

    def fit(self, X_train, y_train,
            X_val=None, y_val=None) -> "LearnSenseBoostVerify":
        """
        Staged training (pipeline contract — no data leakage):
        1. TG-PSF fitted on X_train  → reduced features X_star_train
        2. RGEAB-TE trained on X_star_train
        3. CCSG-PR trained on X_star_train + RGEAB-TE softmax outputs
           (Stages 1 and 2 are frozen before Stage 3 training begins)
        """
        X_train = np.asarray(X_train, dtype=float)

        # Stage 1 — TG-PSF
        X_star_train = self.tgpsf.fit_transform(X_train, y_train)

        # Stage 2 — RGEAB-TE
        self.rgeabte.fit(X_star_train, y_train)
        P_rgeab_train = self.rgeabte.predict_proba(X_star_train)

        # Stage 3 — CCSG-PR (Stages 1 and 2 frozen)
        if X_val is not None:
            X_star_val  = self.tgpsf.transform(np.asarray(X_val, dtype=float))
            P_rgeab_val = self.rgeabte.predict_proba(X_star_val)
            self.ccsgpr.fit(X_star_train, P_rgeab_train, y_train,
                            X_star_val, P_rgeab_val, y_val)
        else:
            self.ccsgpr.fit(X_star_train, P_rgeab_train, y_train)

        return self

    def predict(self, X_test):
        """
        Run full pipeline on test data.

        Returns
        -------
        labels   : (N,)  Predicted class labels (original encoding).
        omega    : (N,)  Uncertainty scores Omega_t (Equation 48).
        accepted : (N,)  Boolean; False = deferred for human review.
        """
        X_test   = np.asarray(X_test,   dtype=float)
        X_star   = self.tgpsf.transform(X_test)
        P_rgeab  = self.rgeabte.predict_proba(X_star)
        labels, omega, accepted = self.ccsgpr.predict(X_star, P_rgeab)
        return labels, omega, accepted

    def predict_stage2(self, X_test):
        """Stage-2 only (RGEAB-TE), 100% coverage — for ablation."""
        X_star  = self.tgpsf.transform(np.asarray(X_test, dtype=float))
        return self.rgeabte.predict(X_star)

    def predict_full_coverage(self, X_test):
        """Stage-3 CCSG-PR with rejection DISABLED — 100% coverage."""
        X_test   = np.asarray(X_test, dtype=float)
        X_star   = self.tgpsf.transform(X_test)
        P_rgeab  = self.rgeabte.predict_proba(X_star)
        # Temporarily disable rejection
        saved_delta     = self.ccsgpr.delta
        self.ccsgpr.delta = 1e9
        labels, omega, _ = self.ccsgpr.predict(X_star, P_rgeab)
        self.ccsgpr.delta = saved_delta
        return labels, omega

    def score_full_coverage(self, X_test, y_test) -> float:
        """Accuracy at 100% coverage — the fair comparator against baselines."""
        labels, _ = self.predict_full_coverage(X_test)
        return float(np.mean(labels == np.asarray(y_test)))

    def score_post_rejection(self, X_test, y_test):
        """
        Returns (post_rejection_accuracy, coverage_rate).
        Post-rejection accuracy is NOT comparable to fully-covered baselines.
        """
        labels, _, accepted = self.predict(X_test)
        y_test = np.asarray(y_test)
        cov    = float(accepted.mean())
        if accepted.sum() == 0:
            return 0.0, cov
        acc = float(np.mean(labels[accepted] == y_test[accepted]))
        return acc, cov

    def feature_reduction_summary(self) -> str:
        n_orig = len(self.tgpsf.similarity_scores_)
        n_kept = len(self.tgpsf.selected_features_)
        return (f"TG-PSF: {n_orig} -> {n_kept} features "
                f"({self.tgpsf.reduction_pct():.1f}% reduction)  "
                f"Ts={self.tgpsf.threshold_Ts_:.4f}")
