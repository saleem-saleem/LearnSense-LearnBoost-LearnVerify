"""
rgeabte.py  -  Regression-Guided Error-Adaptive Boosted Tree Ensemble (Stage 2)
================================================================================
Algorithm 2 of LearnSense-LearnBoost-LearnVerify.

Key design choices
------------------
* Base learner  : DecisionTreeRegressor (max_depth = d_max)
* Regression loss  (Lr_t) couples ensemble to the continuous target.
* Confidence-weighted error (E_t) emphasises uncertain misclassifications.
* Diversity gate  D(h_t, H) discards redundant learners.
"""

from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax


class RGEABTE:
    """
    Regression-Guided Error-Adaptive Boosted Tree Ensemble.

    Parameters
    ----------
    T           : int    Max boosting rounds.           Default 50
    d_max       : int    Max tree depth per learner.    Default 3
    gamma       : float  Regression-coupling weight.    Default 0.5
    xi          : float  Diversity threshold.           Default 0.3
    epsilon     : float  Numerical stability floor.     Default 1e-6
    random_state: int    Seed.                          Default 42
    """

    def __init__(self, T=50, d_max=3, gamma=0.5, xi=0.3,
                 epsilon=1e-6, random_state=42):
        self.T            = T
        self.d_max        = d_max
        self.gamma        = gamma
        self.xi           = xi
        self.epsilon      = epsilon
        self.random_state = random_state

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _softmax_proba(scores: np.ndarray, n_classes: int) -> np.ndarray:
        """Convert raw regression outputs to K-class probabilities."""
        if n_classes == 2:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
            return np.column_stack([1 - p1, p1])
        # multi-class: tile the scalar prediction into K logits
        logits = np.zeros((len(scores), n_classes))
        for k in range(n_classes):
            logits[:, k] = scores - k          # simple offset heuristic
        return softmax(logits, axis=1)

    def _diversity(self, h_outputs: np.ndarray, new_out: np.ndarray) -> float:
        """D(h_t, H) = 1 - mean |corr(h_t, h)| for h in H."""
        if len(h_outputs) == 0:
            return 1.0
        corrs = []
        for prev in h_outputs:
            if np.std(prev) > 1e-9 and np.std(new_out) > 1e-9:
                c = float(np.corrcoef(prev, new_out)[0, 1])
                corrs.append(abs(c))
            else:
                corrs.append(1.0)   # identical  -> not diverse
        return 1.0 - float(np.mean(corrs))

    # ── fit ────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RGEABTE":
        """Train on TRAINING DATA ONLY."""
        X  = np.asarray(X, dtype=float)
        y  = np.asarray(y)
        le = LabelEncoder()
        yc = le.fit_transform(y)              # integer class labels
        self.classes_ = le.classes_
        K  = len(self.classes_)
        N  = len(yc)

        # Continuous target for regression coupling
        y_cont = yc.astype(float)

        w = np.full(N, 1.0 / N)              # instance weights
        self.learners_:  list  = []
        self.alphas_:    list  = []
        h_train_outputs: list  = []           # for diversity check

        rng = np.random.RandomState(self.random_state)

        for t in range(self.T):
            # --- train base learner (weighted) ---------------------------
            ht = DecisionTreeRegressor(
                max_depth=self.d_max,
                random_state=rng.randint(0, 99999))
            ht.fit(X, y_cont, sample_weight=w)

            raw  = ht.predict(X)                     # shape (N,)
            prob = self._softmax_proba(raw, K)        # shape (N, K)
            pred = prob.argmax(axis=1)

            # --- regression loss Lr_t (Equation 24) ----------------------
            Lr = float(np.mean((raw - y_cont) ** 2))
            # normalise to [epsilon, 1-epsilon]
            Lr = np.clip(Lr / (np.var(y_cont) + self.epsilon),
                         self.epsilon, 1 - self.epsilon)

            # --- confidence-weighted classification error E_t (Eq. 27) --
            p_conf     = prob.max(axis=1)             # max prob per instance
            wrong      = (pred != yc).astype(float)
            conf_weight= 1.0 - p_conf                 # high weight for uncertain
            E_t = float(np.sum(w * conf_weight * wrong) /
                        (np.sum(w * conf_weight) + self.epsilon))
            E_t = np.clip(E_t, self.epsilon, 1 - self.epsilon)

            # --- learner weight alpha_t (Equation 28) --------------------
            alpha = (self.gamma * np.log((1 - Lr) / Lr) +
                     (1 - self.gamma) * np.log((1 - E_t) / E_t))

            # --- diversity check (Equations 29-30) -----------------------
            D = self._diversity(h_train_outputs, raw)
            if D < self.xi and t > 0:
                continue                             # discard non-diverse learner

            # --- retain learner ------------------------------------------
            self.learners_.append(ht)
            self.alphas_.append(float(alpha))
            h_train_outputs.append(raw.copy())

            # --- update instance weights (Eq. 31) ------------------------
            correct = (pred == yc).astype(float)
            w *= np.exp(-alpha * (2 * correct - 1))
            w  = np.clip(w, 1e-9, None)
            w /= w.sum()

        self.alphas_ = np.array(self.alphas_)
        return self

    # ── predict ────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability estimate."""
        X  = np.asarray(X, dtype=float)
        K  = len(self.classes_)
        P  = np.zeros((len(X), K))
        total_alpha = self.alphas_.sum() + 1e-9
        for ht, a in zip(self.learners_, self.alphas_):
            raw  = ht.predict(X)
            prob = self._softmax_proba(raw, K)
            P   += (a / total_alpha) * prob
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (original encoding)."""
        idx = self.predict_proba(X).argmax(axis=1)
        return self.classes_[idx]

    def __repr__(self):
        n = len(getattr(self, "learners_", []))
        return (f"RGEABTE(T={self.T}, d_max={self.d_max}, gamma={self.gamma}, "
                f"xi={self.xi}, n_retained={n})")
