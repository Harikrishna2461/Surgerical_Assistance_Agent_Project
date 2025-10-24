import numpy as np
import tensorflow as tf

PHASE_MAP = {
    0: "preparation",
    1: "carlot-triangle-dissection",
    2: "clipping-and-cutting",
    3: "gallbladder-dissection",
    4: "gallbladder-packaging",
    5: "cleaning-and-coagulation",
    6: "gallbladder-extraction",
}

def predict_phase_from_image(image: np.ndarray, model_path: str) -> str:
    """
    Single-label surgical phase recognition (returns exactly one phase string).

    Args:
        image: np.ndarray (H, W, 3), already resized/preprocessed to match the model.
        model_path: full path to your .keras model.

    Returns:
        phase label (str), e.g. "clipping-and-cutting".
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be (H, W, 3)")

    # Load model inside the function (per request)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Forward pass
    x = image.astype("float32")[None, ...]   # (1, H, W, 3)
    y = model.predict(x, verbose=0)

    # Squeeze to (C,)
    if y.ndim == 2:        # (1, C)
        y = y[0]
    elif y.ndim == 3:      # (1, T, C) -> average over time
        y = y.mean(axis=1)[0]
    else:
        y = y.reshape(-1)

    # Ensure probabilities (softmax if logits or not normalized)
    y = y.astype("float32")
    if (y.min() < 0.0) or (y.max() > 1.0) or (abs(y.sum() - 1.0) > 1e-3):
        y = tf.nn.softmax(y).numpy()

    idx = int(np.argmax(y))
    return PHASE_MAP.get(idx, f"UNKNOWN_{idx}")