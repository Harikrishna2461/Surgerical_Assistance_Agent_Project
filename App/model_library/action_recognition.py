import numpy as np
import tensorflow as tf

VERB_MAP_COMPACT = {
    0: "grasp",
    1: "retract",
    2: "dissect",
    3: "coagulate",
    4: "clip",
    5: "cut",
    6: "aspirate",
    7: "irrigate",
    8: "null_verb",
}

def predict_action_verb_from_image(image: np.ndarray, model_path: str) -> str:
    """
    Single-label action recognition (returns exactly one verb).
    - Loads model inside this function (as requested).
    - Input `image` should already be resized/preprocessed to the model's expected format.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be (H,W,3)")

    # load model per call
    model = tf.keras.models.load_model(model_path, compile=False)

    # forward
    x = image.astype("float32")[None, ...]   # (1,H,W,3)
    y = model.predict(x, verbose=0)

    # squeeze to (C,)
    if y.ndim == 2:      # (1,C)
        y = y[0]
    elif y.ndim == 3:    # (1,T,C) -> average over time to (C,)
        y = y.mean(axis=1)[0]
    else:
        y = y.reshape(-1)

    # ensure probabilities; use softmax then argmax for single label
    y = y.astype("float32")
    if (y.min() < 0.0) or (y.max() > 1.0) or (abs(y.sum() - 1.0) > 1e-3):
        y = tf.nn.softmax(y).numpy()

    idx = int(np.argmax(y))
    return VERB_MAP_COMPACT.get(idx, f"UNKNOWN_{idx}")
