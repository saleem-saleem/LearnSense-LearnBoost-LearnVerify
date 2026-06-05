"""
baselines.py  -  All eight baseline classifiers with unified interface
======================================================================
Baselines are fitted and evaluated on TG-PSF-selected features
(same as RGEAB-TE and CCSG-PR) unless noted.
"""

from __future__ import annotations
import numpy as np
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier,
                                   AdaBoostClassifier,
                                   GradientBoostingClassifier)
from xgboost               import XGBClassifier


def get_tree_baselines(random_state=42) -> dict:
    """Return the five tree/ensemble baselines."""
    return {
        "Single_DT": DecisionTreeClassifier(
            max_depth=3, random_state=random_state),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=50, learning_rate=1.0, random_state=random_state),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state, verbosity=0),
    }


# ── Deep learning baselines (require TensorFlow) ─────────────────────────
def build_lstm_baseline(n_features: int, n_classes: int,
                        hidden: int = 64, random_state: int = 42):
    """
    LSTM baseline: treats each feature as one time step.
    Architecture: LSTM(64) -> LSTM(32) -> Dense(K, softmax)
    Params: ~120,258 (fixed, all datasets)
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
        inp  = tf.keras.Input(shape=(n_features, 1))
        x    = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
        x    = tf.keras.layers.LSTM(32)(x)
        out  = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    except ImportError:
        raise ImportError("TensorFlow required for LSTM baseline. "
                          "pip install tensorflow")


def build_transformer_baseline(n_features: int, n_classes: int,
                                d_model: int = 32, n_heads: int = 2,
                                random_state: int = 42):
    """
    Transformer baseline: each feature as one token.
    Architecture: Embedding -> 2x TransformerBlock -> GlobalAvgPool -> Dense(K)
    Params: ~20,000 (varies with n_features)
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
        inp   = tf.keras.Input(shape=(n_features,))
        x     = tf.keras.layers.Dense(d_model)(tf.expand_dims(inp, -1))
        for _ in range(2):
            attn  = tf.keras.layers.MultiHeadAttention(
                num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
            x     = tf.keras.layers.LayerNormalization()(x + attn)
            ff    = tf.keras.layers.Dense(d_model * 2, activation="relu")(x)
            ff    = tf.keras.layers.Dense(d_model)(ff)
            x     = tf.keras.layers.LayerNormalization()(x + ff)
        x     = tf.keras.layers.GlobalAveragePooling1D()(x)
        out   = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    except ImportError:
        raise ImportError("TensorFlow required for Transformer baseline.")


def build_bte_baseline(n_features: int, n_classes: int,
                       hidden: int = 64, random_state: int = 42):
    """
    BTE (BERT-style Tabular Encoder): bidirectional transformer on tabular tokens.
    NOT a pretrained NLP BERT. Trained from scratch on numerical feature vectors.
    Params: ~72,000-76,000 (varies with n_features, n_classes)
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
        inp  = tf.keras.Input(shape=(n_features,))
        x    = tf.keras.layers.Dense(hidden, activation="relu")(
               tf.expand_dims(inp, -1))
        # Bidirectional attention (simulating BERT-style)
        fwd  = tf.keras.layers.LSTM(hidden // 2, return_sequences=True)(x)
        bwd  = tf.keras.layers.LSTM(hidden // 2, return_sequences=True,
                                     go_backwards=True)(x)
        x    = tf.keras.layers.Concatenate()([fwd, bwd])
        x    = tf.keras.layers.GlobalAveragePooling1D()(x)
        x    = tf.keras.layers.Dense(hidden, activation="relu")(x)
        out  = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    except ImportError:
        raise ImportError("TensorFlow required for BTE baseline.")


def fit_dl_baseline(model, X_train, y_train, X_val=None, y_val=None,
                    epochs=200, batch_size=32, patience=20):
    """
    Fit a TF/Keras deep learning baseline model.
    Early stopping on val_accuracy; restore best weights.
    """
    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy" if X_val is not None else "accuracy",
            patience=patience,
            restore_best_weights=True,
            mode="max")
    ]
    val_data = (X_val, y_val) if X_val is not None else None
    model.fit(X_train, y_train,
              validation_data=val_data,
              epochs=epochs, batch_size=batch_size,
              callbacks=callbacks, verbose=0)
    return model
