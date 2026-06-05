"""
ccsgpr.py  -  Confidence-Calibrated Similarity-Guided Prediction Refinement (Stage 3)
======================================================================================
Algorithm 3 of LearnSense-LearnBoost-LearnVerify.

Architecture note
-----------------
CCSG-PR is NOT a recurrent model.  It processes each tabular instance
with ONE LSTM-style gate step (C_0 = h_0 = 0).  The gating structure
is borrowed to enable phi_t-modulated information routing.

Training
--------
Stage 1 (TG-PSF) and Stage 2 (RGEAB-TE) must be FROZEN before fitting
CCSG-PR.  Pass their outputs as pre-computed arrays.

Dependencies
------------
numpy, scipy  (no TensorFlow required; pure numpy implementation)
"""

from __future__ import annotations
import numpy as np
from scipy.special import softmax


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class CCSGPR:
    """
    Confidence-Calibrated Similarity-Guided Prediction Refinement.

    Parameters
    ----------
    d_h         : int    Hidden state dimension.            Default 64
    k           : float  Relevance-score decay factor.      Default 0.5
    tau_r       : float  Feature-activation threshold.      Default 0.6
    delta       : float  Uncertainty acceptance threshold.  Default 0.10
    xi          : float  Min loss for selective update.     Default 0.01
    eta         : float  Adam learning rate.                Default 0.001
    epochs      : int    Max training epochs.               Default 200
    patience    : int    Early-stopping patience.           Default 15
    random_state: int    Seed.                              Default 42
    """

    def __init__(self, d_h=64, k=0.5, tau_r=0.6, delta=0.10, xi=0.01,
                 eta=0.001, epochs=200, patience=15, random_state=42):
        self.d_h          = d_h
        self.k            = k
        self.tau_r        = tau_r
        self.delta        = delta
        self.xi           = xi
        self.eta          = eta
        self.epochs       = epochs
        self.patience     = patience
        self.random_state = random_state

    # ── Internal helpers ──────────────────────────────────────────────
    def _xavier(self, rows, cols, rng):
        limit = np.sqrt(6.0 / (rows + cols))
        return rng.uniform(-limit, limit, (rows, cols))

    def _init_weights(self, d_in, K, rng):
        dh = self.d_h
        W = {
            "Wi": self._xavier(dh, d_in, rng),
            "Wf": self._xavier(dh, d_in, rng),
            "Wc": self._xavier(dh, d_in, rng),
            "Wo": self._xavier(dh, d_in, rng),
            "Wy": self._xavier(K,  dh,   rng),
            "bi": np.zeros(dh),
            "bf": np.zeros(dh),
            "bc": np.zeros(dh),
            "bo": np.zeros(dh),
            "by": np.zeros(K),
        }
        # Adam moment estimates
        M = {k: np.zeros_like(v) for k, v in W.items()}
        V = {k: np.zeros_like(v) for k, v in W.items()}
        return W, M, V

    def _adam_step(self, W, M, V, grads, t, beta1=0.9, beta2=0.999, eps=1e-7):
        for key in W:
            M[key] = beta1 * M[key] + (1 - beta1) * grads[key]
            V[key] = beta2 * V[key] + (1 - beta2) * grads[key] ** 2
            mhat   = M[key] / (1 - beta1 ** t)
            vhat   = V[key] / (1 - beta2 ** t)
            W[key] -= self.eta * mhat / (np.sqrt(vhat) + eps)

    # ── Class prototypes ─────────────────────────────────────────────
    def _compute_prototypes(self, X_star, y_int):
        C = {}
        for c in range(self.K_):
            mask = (y_int == c)
            C[c] = X_star[mask].mean(axis=0) if mask.any() else np.zeros(X_star.shape[1])
        return C

    # ── Per-instance forward pass ────────────────────────────────────
    def _forward(self, x_star, p_rgeab, y_hat, W):
        """Returns: y_refined, Z_tm, phi_t, h_t, gate_cache"""
        # Step 1 — relevance (Eq. 37-39)
        proto  = self.prototypes_[int(y_hat)]
        d_j    = np.abs(x_star - proto)
        R_i    = float(np.exp(-self.k * d_j.min()))
        alpha  = x_star if R_i >= self.tau_r else np.zeros_like(x_star)

        # Step 2 — class similarity (Eq. 40)
        na, np_ = np.linalg.norm(alpha), np.linalg.norm(proto)
        Z_tm   = float(alpha @ proto) / (na * np_ + 1e-9) if (na > 1e-9 and np_ > 1e-9) else 0.0

        # Step 3 — calibration discrepancy phi_t (Eq. 41)
        phi_t  = float(Z_tm - p_rgeab[int(y_hat)])   # scalar; NOTE: distinct from c_i

        # Step 4 — fuse and normalise (Eq. 35-36)
        fused  = np.concatenate([alpha, p_rgeab])
        x_t    = (fused - self.mu_) / self.sigma_

        # Step 5 — single LSTM step; C_0=0, h_0=0 (Eq. 42-46)
        i_g = _sigmoid(W["Wi"] @ x_t + W["bi"] - phi_t)
        f_g = _sigmoid(W["Wf"] @ x_t + W["bf"] - phi_t)
        g_g = np.tanh( W["Wc"] @ x_t + W["bc"])
        o_g = _sigmoid(W["Wo"] @ x_t + W["bo"])
        C_t = f_g * 0 + i_g * g_g               # C_prev = 0
        h_t = o_g * np.tanh(C_t)

        # Step 6 — output (Eq. 47)
        y_ref = W["Wy"] @ h_t + W["by"]

        cache = dict(x_t=x_t, i_g=i_g, f_g=f_g, g_g=g_g,
                     o_g=o_g, C_t=C_t, h_t=h_t, phi_t=phi_t)
        return y_ref, Z_tm, phi_t, h_t, cache

    def _cross_entropy(self, logits, y_true_int):
        probs = softmax(logits)
        return -float(np.log(probs[y_true_int] + 1e-12))

    def _backward(self, y_true_int, y_ref, cache, W):
        """Manual backprop for single LSTM step."""
        K, dh  = len(W["by"]), self.d_h
        probs  = softmax(y_ref)
        dlogit = probs.copy()
        dlogit[y_true_int] -= 1.0                # (K,)

        grads  = {}
        grads["Wy"] = np.outer(dlogit, cache["h_t"])
        grads["by"] = dlogit

        dh_t   = W["Wy"].T @ dlogit             # (dh,)
        dC_t   = dh_t * cache["o_g"] * (1 - np.tanh(cache["C_t"])**2)

        do_g   = dh_t * np.tanh(cache["C_t"])
        di_g   = dC_t * cache["g_g"]
        dg_g   = dC_t * cache["i_g"]
        df_g   = dC_t * 0                       # C_prev=0, so df_g negligible

        do_pre = do_g * cache["o_g"] * (1 - cache["o_g"])
        di_pre = di_g * cache["i_g"] * (1 - cache["i_g"])
        dg_pre = dg_g * (1 - cache["g_g"]**2)
        df_pre = df_g * cache["f_g"] * (1 - cache["f_g"])

        grads["Wo"] = np.outer(do_pre, cache["x_t"])
        grads["bo"] = do_pre
        grads["Wi"] = np.outer(di_pre, cache["x_t"])
        grads["bi"] = di_pre
        grads["Wc"] = np.outer(dg_pre, cache["x_t"])
        grads["bc"] = dg_pre
        grads["Wf"] = np.outer(df_pre, cache["x_t"])
        grads["bf"] = df_pre
        return grads

    # ── Fit ───────────────────────────────────────────────────────────
    def fit(
        self,
        X_star_train:  np.ndarray,
        P_rgeab_train: np.ndarray,
        y_train:       np.ndarray,
        X_star_val:    np.ndarray | None = None,
        P_rgeab_val:   np.ndarray | None = None,
        y_val:         np.ndarray | None = None,
    ) -> "CCSGPR":
        """
        Train CCSG-PR.  Stage-1 and Stage-2 outputs must be pre-computed.

        Parameters
        ----------
        X_star_train  : (N_tr, |A*|)  TG-PSF-selected training features.
        P_rgeab_train : (N_tr, K)     Stage-2 softmax probabilities (training).
        y_train       : (N_tr,)       True class labels.
        X_star_val    : (N_val, |A*|) [optional] for early stopping.
        P_rgeab_val   : (N_val, K)    [optional]
        y_val         : (N_val,)      [optional]
        """
        X_star_train  = np.asarray(X_star_train,  dtype=float)
        P_rgeab_train = np.asarray(P_rgeab_train, dtype=float)
        y_train       = np.asarray(y_train)

        from sklearn.preprocessing import LabelEncoder
        le              = LabelEncoder()
        y_int           = le.fit_transform(y_train)
        self.classes_   = le.classes_
        self.K_         = len(self.classes_)
        K, N            = self.K_, len(y_int)

        # Prototypes (training instances only)
        self.prototypes_ = self._compute_prototypes(X_star_train, y_int)

        # Normalisation stats (training set only)
        fused          = np.hstack([X_star_train, P_rgeab_train])
        self.mu_       = fused.mean(axis=0)
        self.sigma_    = fused.std(axis=0) + 1e-8
        d_in           = fused.shape[1]

        rng = np.random.RandomState(self.random_state)
        W, M, V = self._init_weights(d_in, K, rng)

        y_hat_train = P_rgeab_train.argmax(axis=1)

        best_val_acc  = -1.0
        patience_cnt  = 0
        best_W        = {k: v.copy() for k, v in W.items()}
        adam_t        = 0

        use_val = (X_star_val is not None)
        if use_val:
            X_star_val  = np.asarray(X_star_val,  dtype=float)
            P_rgeab_val = np.asarray(P_rgeab_val, dtype=float)
            y_val_int   = le.transform(y_val)

        for epoch in range(self.epochs):
            idx = rng.permutation(N)
            for i in idx:
                y_ref, Z_tm, phi_t, h_t, cache = self._forward(
                    X_star_train[i], P_rgeab_train[i], y_hat_train[i], W)
                loss = self._cross_entropy(y_ref, y_int[i])
                if loss < self.xi:
                    continue                       # selective update (Eq. 52)
                adam_t += 1
                grads = self._backward(y_int[i], y_ref, cache, W)
                self._adam_step(W, M, V, grads, adam_t)

            # Early stopping
            if use_val:
                preds, _, _ = self._predict_all(X_star_val, P_rgeab_val,
                                                P_rgeab_val.argmax(axis=1), W)
                val_acc = float(np.mean(le.inverse_transform(preds) == y_val))
            else:
                preds, _, _ = self._predict_all(X_star_train, P_rgeab_train,
                                                y_hat_train, W)
                val_acc = float(np.mean(preds == y_int))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_W       = {k: v.copy() for k, v in W.items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        self.W_          = best_W
        self.best_val_acc_ = best_val_acc
        return self

    # ── Predict (internal) ────────────────────────────────────────────
    def _predict_all(self, X_star, P_rgeab, y_hat_arr, W):
        preds    = np.zeros(len(X_star), dtype=int)
        omega    = np.zeros(len(X_star))
        accepted = np.ones(len(X_star),  dtype=bool)
        for i in range(len(X_star)):
            y_ref, Z_tm, phi_t, _, _ = self._forward(
                X_star[i], P_rgeab[i], y_hat_arr[i], W)
            probs   = softmax(y_ref)
            c_i     = float(probs.max())
            Omega_t = abs(Z_tm - c_i)           # Equation 48
            preds[i]    = int(probs.argmax())
            omega[i]    = Omega_t
            accepted[i] = Omega_t <= self.delta
        return preds, omega, accepted

    def predict(self, X_star, P_rgeab, return_uncertainty=False):
        """
        Predict with selective rejection.

        Returns
        -------
        labels      : np.ndarray (N,)   Predicted class labels.
                      Deferred instances get the best-available label.
        omega       : np.ndarray (N,)   Uncertainty score Omega_t.
        accepted    : np.ndarray (N,)   Boolean mask; False = deferred.
        """
        X_star  = np.asarray(X_star,  dtype=float)
        P_rgeab = np.asarray(P_rgeab, dtype=float)
        y_hat   = P_rgeab.argmax(axis=1)
        preds, omega, accepted = self._predict_all(X_star, P_rgeab, y_hat, self.W_)
        labels  = self.classes_[preds]
        if return_uncertainty:
            return labels, omega, accepted
        return labels, omega, accepted

    def __repr__(self):
        return (f"CCSGPR(d_h={self.d_h}, k={self.k}, tau_r={self.tau_r}, "
                f"delta={self.delta}, eta={self.eta})")
