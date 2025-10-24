import numpy as np
import tensorflow as tf



def predict_vitals_from_stft_array(
    stft_array: np.ndarray,
    model_path: str,
    y_mean: np.ndarray | None = None,  # shape (3,) if you want standard units
    y_std:  np.ndarray | None = None,  # shape (3,) if you want standard units
):
    """
    Predict 3 vitals from STFT tiles. Model is loaded INSIDE this function.

    Args
    stft_array : (F,T) or (N,F,T) float32
    model_path : path to your .keras model
    y_mean/y_std: OPTIONAL train-set stats to convert z-scores -> real units
                    (order must be [MAP, SpO2, ETCO2])

    Returns
    If single tile: {'MAP_mmHg': float, 'SpO2_pct': float, 'ETCO2_mmHg': float}
    If batch:       np.ndarray (N,3) in order [MAP, SpO2, ETCO2]
    """
    model = tf.keras.models.load_model(model_path, compile=False)

    X = np.asarray(stft_array, dtype=np.float32)
    single = X.ndim == 2
    if single:
        X = X[None, ...]
    elif X.ndim != 3:
        raise ValueError("stft_array must be (F,T) or (N,F,T)")
    X = X[..., None]  # (N,F,T,1)

    y = model.predict(X, verbose=0).astype(np.float32)  # (N,3)

    # If stats provided, convert z-scores -> real units
    if (y_mean is not None) and (y_std is not None):
        y = y * y_std[None, :] + y_mean[None, :]

    if single:
        return {
            "MAP_mmHg":  float(y[0, 0]),
            "SpO2_pct":  float(y[0, 1]),
            "ETCO2_mmHg":float(y[0, 2]),
        }
    return y