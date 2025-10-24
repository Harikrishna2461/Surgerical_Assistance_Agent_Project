import os
import cv2
import numpy as np
import tempfile
from typing import Union, BinaryIO, Tuple, Optional

def _to_tmp_video(file_or_bytes: Union[str, bytes, BinaryIO]) -> str:
    """
    Ensure we have a real file path that OpenCV can read.
    Accepts:
    - streamlit UploadedFile (has .read()/.getvalue())
    - raw bytes
    - str path
    Returns a temp file path.
    """
    if isinstance(file_or_bytes, str) and os.path.exists(file_or_bytes):
        return file_or_bytes

    if hasattr(file_or_bytes, "read"):
        data = file_or_bytes.read()
    elif hasattr(file_or_bytes, "getvalue"):
        data = file_or_bytes.getvalue()
    elif isinstance(file_or_bytes, (bytes, bytearray)):
        data = file_or_bytes
    else:
        raise ValueError("Unsupported input: pass a file path, bytes, or a file-like object")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(data); tmp.flush(); tmp.close()
    return tmp.name

def _center_crop_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        s = (h - w) // 2
        return img[s:s+w, :, :]
    else:
        s = (w - h) // 2
        return img[:, s:s+h, :]

def extract_frames_rgb(
    video: Union[str, bytes, BinaryIO],
    every_n_seconds: float = 1.0,            # 1 FPS like CholecT50
    target_size: Tuple[int, int] = (224, 224),
    crop_square: bool = True,
    max_frames: Optional[int] = None
) -> np.ndarray:
    """
    Extract RGB frames at ~1 FPS, center-crop to square (optional), resize to target_size,
    and return (N, H, W, 3) uint8 with NO normalization/preprocessing.

    Args:
    video: streamlit file obj, raw bytes, or a string path.
    every_n_seconds: sampling interval; 1.0 ≈ 1 FPS.
    target_size: (W, H) to resize each frame (model will do its own preprocessing).
    crop_square: center-crop to square before resize (helps with endoscopic FOV).
    max_frames: optional cap.

    Returns:
    frames_np: np.ndarray shape (N, H, W, 3), dtype=uint8, RGB order.
    """
    tmp_path = _to_tmp_video(video)
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video. Ensure it's a valid video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 25.0  # fallback

    stride = max(int(round(fps * every_n_seconds)), 1)

    frames = []
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            # BGR -> RGB
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # optional square crop, then resize
            if crop_square:
                rgb = _center_crop_to_square(rgb)
            rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
            frames.append(rgb.astype(np.uint8))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1

    cap.release()
    try:
        if tmp_path and not isinstance(video, str):
            os.remove(tmp_path)
    except Exception:
        pass

    if not frames:
        raise RuntimeError("No frames extracted. Try a different codec or reduce every_n_seconds.")

    return np.stack(frames, axis=0)