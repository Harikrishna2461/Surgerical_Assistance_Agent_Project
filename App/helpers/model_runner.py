import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import traceback, sys
# App/helpers/model_runner.py (very top)
from model_library.instrument_recognition import predict_instruments_from_image
from model_library.action_recognition import predict_action_verb_from_image
from model_library.phase_recognition import predict_phase_from_image
from model_library.target_organ_recognition import predict_targets_from_image
from model_library.vitals_recognition import predict_vitals_from_stft_array

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[model_runner] {fn.__name__} failed:", e, file=sys.stderr)
        traceback.print_exc()
        return e

def _predict_one_frame(
    image: np.ndarray,
    stft_tile: Optional[np.ndarray],
    paths: Dict[str, str],
    instrument_threshold: float,
    target_threshold: float,
    return_top1_if_none: bool,
    y_mean: Optional[np.ndarray],
    y_std:  Optional[np.ndarray],
) -> Dict[str, str]:
    inst = _safe(
        predict_instruments_from_image,
        image=image, model_path=paths["instrument"],
        threshold=instrument_threshold, per_class_thresholds=None,
        return_top1_if_none=return_top1_if_none,
    )
    instrument_str = "" if isinstance(inst, Exception) else (
        ", ".join(inst) if isinstance(inst, (list, tuple)) else str(inst)
    )

    act = _safe(predict_action_verb_from_image, image=image, model_path=paths["action"])
    action_str = "" if isinstance(act, Exception) else act

    ph = _safe(predict_phase_from_image, image=image, model_path=paths["phase"])
    phase_str = "" if isinstance(ph, Exception) else ph

    tgt = _safe(
        predict_targets_from_image,
        image=image, model_path=paths["target"],
        threshold=target_threshold, per_class_thresholds=None,
        return_top1_if_none=return_top1_if_none,
    )
    target_str = "" if isinstance(tgt, Exception) else (
        ", ".join(tgt) if isinstance(tgt, (list, tuple)) else str(tgt)
    )

    # Vitals (proper scale + names)
    if stft_tile is None:
        map_str = spo2_str = etco2_str = "NaN"
    else:
        vit = _safe(
            predict_vitals_from_stft_array,
            stft_array=stft_tile,
            model_path=paths["vitals"],
            y_mean=y_mean, y_std=y_std
        )
        if isinstance(vit, Exception):
            map_str = spo2_str = etco2_str = "NaN"
        elif isinstance(vit, dict):
            MAP  = vit.get("MAP_mmHg", np.nan)
            SPO2 = vit.get("SpO2_pct", np.nan)
            ET   = vit.get("ETCO2_mmHg", np.nan)
            map_str  = f"{float(MAP):.2f}"  if np.isfinite(MAP) else "NaN"
            spo2_str = f"{float(SPO2):.2f}" if np.isfinite(SPO2) else "NaN"
            etco2_str= f"{float(ET):.2f}"   if np.isfinite(ET) else "NaN"
        elif isinstance(vit, np.ndarray) and vit.ndim == 2 and vit.shape[1] == 3:
            map_str  = f"{float(vit[0,0]):.2f}"
            spo2_str = f"{float(vit[0,1]):.2f}"
            etco2_str= f"{float(vit[0,2]):.2f}"
        else:
            map_str = spo2_str = etco2_str = "NaN"

    return {
        "Instrument": instrument_str,
        "Phase": phase_str,
        "Target": target_str,
        "Surgical_Action": action_str,
        "MAP_mmHg":  map_str,
        "SpO2_pct":  spo2_str,
        "ETCO2_mmHg":etco2_str,
    }

def run_per_frame_predictions(
    frames: np.ndarray,                # (N,H,W,3) RGB uint8
    stft_array: Optional[np.ndarray],  # (M,F,T) or (F,T) or None
    paths: Dict[str, str],             # model paths dict
    *,
    instrument_threshold: float = 0.5,
    target_threshold: float = 0.5,
    return_top1_if_none: bool = True,
    y_mean: Optional[np.ndarray] = None,  # (3,) if your vitals model outputs z-scores
    y_std:  Optional[np.ndarray] = None,  # (3,)
    max_workers: int = 8
) -> List[Dict[str, str]]:
    """
    Run all five predictions for each frame (in parallel) and return a list of JSON rows,
    with vitals reported as MAP_mmHg / SpO2_pct / ETCO2_mmHg in proper scales when y_mean/y_std provided.

    Notes:
      - If stft_array is (F,T), it is broadcast to all frames.
      - If stft_array is (M,F,T) and M != N, we use min(N,M) and pad last.
      - If stft_array is None, vitals entries are "NaN".
    """
    assert isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3

    N = frames.shape[0]
    if stft_array is None:
        tiles = [None] * N
    else:
        S = np.asarray(stft_array)
        if S.ndim == 2:
            tiles = [S for _ in range(N)]
        elif S.ndim == 3:
            M = S.shape[0]
            K = min(N, M)
            tiles = [S[i] for i in range(K)] + [S[M-1]]*(N-K)
        else:
            raise ValueError("stft_array must be None, (F,T) or (M,F,T)")

    out: List[Dict[str, str]] = [None] * N
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                _predict_one_frame,
                frames[i], tiles[i], paths,
                instrument_threshold, target_threshold, return_top1_if_none,
                y_mean, y_std
            ): i for i in range(N)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception:
                out[i] = {
                    "Instrument": "",
                    "Phase": "",
                    "Target": "",
                    "Surgical_Action": "",
                    "MAP_mmHg":  "NaN",
                    "SpO2_pct":  "NaN",
                    "ETCO2_mmHg":"NaN",
                }
    return out
