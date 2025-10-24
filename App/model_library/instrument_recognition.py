import numpy as np
import tensorflow as tf

INSTRUMENT_MAP = {
    0: "grasper",
    1: "bipolar",
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
}
CLASS_NAMES = [INSTRUMENT_MAP[i] for i in range(len(INSTRUMENT_MAP))]

def predict_instruments_from_image(
    image: np.ndarray,
    model_path: str,
    threshold: float = 0.5,
    per_class_thresholds: dict | None = None,
    return_top1_if_none: bool = True,
) -> list[str]:
    """
    Multi-label surgical tool recognition.
    Loads the model inside this function as requested.

    Args:
    image: np.ndarray (H,W,3), already resized/preprocessed for the model.
    model_path: path to your .keras model.
    Returns:
    List[str] like ['grasper','scissors'].
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be (H,W,3)")

    # load model (inside)
    model = tf.keras.models.load_model(model_path, compile=False)

    x = image.astype("float32")[None, ...]  # (1,H,W,3)
    y = model.predict(x, verbose=0)

    # squeeze to (C,)
    if y.ndim == 2:              # (1,C)
        y = y[0]
    elif y.ndim == 3:            # (1,T,C) -> avg over time
        y = y.mean(axis=1)[0]
    else:
        y = y.reshape(-1)

    # ensure probabilities (apply sigmoid if logits)
    probs = y.astype("float32")
    if (probs.min() < 0.0) or (probs.max() > 1.0):
        probs = tf.math.sigmoid(probs).numpy()

    # thresholds (global or per-class)
    if per_class_thresholds:
        thr = np.array([
            per_class_thresholds.get(i, per_class_thresholds.get(CLASS_NAMES[i], threshold))
            for i in range(len(CLASS_NAMES))
        ], dtype="float32")
    else:
        thr = np.full_like(probs, float(threshold))

    sel = probs >= thr
    if return_top1_if_none and not np.any(sel):
        sel[np.argmax(probs)] = True

    return [CLASS_NAMES[i] for i in np.where(sel)[0].tolist()]