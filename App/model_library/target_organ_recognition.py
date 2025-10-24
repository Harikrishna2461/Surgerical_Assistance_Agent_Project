import numpy as np
import tensorflow as tf

# Model output index (0..12) -> final target name (remapped)
MODEL_TARGET_NAMES = {
    0:  "gallbladder",
    1:  "cystic_plate",
    2:  "cystic_duct",
    3:  "cystic_artery",
    4:  "cystic_pedicle",
    5:  "blood_vessel",
    6:  "fluid",
    7:  "abdominal_wall_cavity",
    8:  "liver",
    9:  "omentum",        # remapped from original index 10
    10: "gut",            # remapped from original index 12
    11: "specimen_bag",   # remapped from original index 13
    12: "null_target",    # remapped from original index 14
}

def predict_targets_from_image(
    image: np.ndarray,
    model_path: str,
    threshold: float = 0.5,
    per_class_thresholds: dict | None = None,
    return_top1_if_none: bool = True,
) -> list[str]:
    """
    Multi-label Target recognition.
    Loads the model inside this function and returns only the list of target names.

    Args:
      image: np.ndarray (H,W,3), already resized/preprocessed for the model.
      model_path: path to your .keras model trained with the 13-class remap.
      threshold: global probability cutoff (used if per_class_thresholds is None).
      per_class_thresholds: optional {idx or name: thr} to customize per-class cutoffs.
      return_top1_if_none: if nothing meets threshold(s), return the top-1 class.

    Returns:
      List[str], e.g. ["gallbladder", "cystic_duct"].
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be (H,W,3)")

    # load model per call
    model = tf.keras.models.load_model(model_path, compile=False)

    # forward
    x = image.astype("float32")[None, ...]   # (1,H,W,3)
    y = model.predict(x, verbose=0)

    # squeeze to (C,)
    if y.ndim == 2:          # (1,C)
        y = y[0]
    elif y.ndim == 3:        # (1,T,C) -> average over time
        y = y.mean(axis=1)[0]
    else:
        y = y.reshape(-1)

    # ensure probabilities for multi-label (sigmoid)
    probs = y.astype("float32")
    if (probs.min() < 0.0) or (probs.max() > 1.0):
        probs = tf.math.sigmoid(probs).numpy()

    # thresholds
    if per_class_thresholds:
        thr_vec = np.array([
            per_class_thresholds.get(i,
                per_class_thresholds.get(MODEL_TARGET_NAMES[i], threshold))
            for i in range(len(MODEL_TARGET_NAMES))
        ], dtype="float32")
    else:
        thr_vec = np.full_like(probs, float(threshold))

    sel = probs >= thr_vec
    if return_top1_if_none and not np.any(sel):
        sel[np.argmax(probs)] = True

    return [MODEL_TARGET_NAMES[i] for i in np.where(sel)[0].tolist()]