import io, os
import numpy as np
from typing import Union, BinaryIO, Tuple, Optional, List

from scipy.signal import butter, filtfilt, find_peaks, stft

# ----------------- defaults (match your training) -----------------
FS         = 100.0        # Hz (target resample rate)
WIN_PRE    = 0.30         # seconds before R
WIN_POST   = 0.60         # seconds after R
RR_MINMAX  = (0.35, 1.8)  # valid RR in seconds
BP_LOHI    = (0.5, 40.0)  # ECG bandpass (Hz)

# STFT params (time–frequency tiles used by your model)
NPERSEG    = 24
NOVERLAP   = 20
NFFT       = 128
FMAX       = 45.0         # Hz

# Target spectrogram size for the vitals model
OUT_FBINS  = 58
OUT_TBINS  = 17
# ------------------------------------------------------------------

# ----------------- CSV loading without pandas -----------------
def _read_all_bytes(file_obj: Union[str, bytes, BinaryIO]) -> bytes:
    if isinstance(file_obj, str) and os.path.exists(file_obj):
        with open(file_obj, "rb") as f:
            return f.read()
    if hasattr(file_obj, "read"):
        # Streamlit UploadedFile etc.
        return file_obj.read()
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    raise ValueError("Pass a file path, bytes, or a file-like object for the ECG CSV.")

def _sniff_delim(head: bytes) -> str:
    """
    Very light delimiter sniffer: prefer ','; fallback to ';' or '\t'.
    """
    s = head.decode("utf-8", errors="ignore")
    comma = s.count(",")
    semi  = s.count(";")
    tab   = s.count("\t")
    if comma >= semi and comma >= tab and comma > 0: return ","
    if semi  >= tab and semi  > 0: return ";"
    if tab   > 0: return "\t"
    return ","  # default

def _genfromtxt_with_names(buf: bytes, delimiter: str):
    """
    Try names=True first (header row). If it fails, retry names=False (no header).
    Returns (names_list_or_None, array_2d_float).
    """
    bio = io.BytesIO(buf)
    try:
        arr = np.genfromtxt(bio, delimiter=delimiter, names=True, dtype=None, encoding="utf-8")
        if arr.size == 0:
            raise ValueError("Empty CSV")
        # Convert structured array -> 2D float
        names = list(arr.dtype.names)
        data  = np.vstack([np.asarray(arr[n], dtype=float) for n in names]).T
        return names, data
    except Exception:
        # retry: no header
        bio2 = io.BytesIO(buf)
        arr2 = np.genfromtxt(bio2, delimiter=delimiter, dtype=float)
        if arr2.ndim == 1:
            arr2 = arr2.reshape(-1, 1)
        cols = arr2.shape[1] if arr2.size else 0
        names = [f"col{i}" for i in range(cols)] if cols else None
        return names, arr2

def _load_ecg_csv_np(file_obj: Union[str, bytes, BinaryIO]) -> Tuple[Optional[List[str]], np.ndarray, str]:
    """
    Returns (names_or_None, data_2d_float, delimiter_used)
    """
    buf = _read_all_bytes(file_obj)
    # Peek first 4KB to sniff delimiter
    delim = _sniff_delim(buf[:4096])
    names, data = _genfromtxt_with_names(buf, delimiter=delim)
    if data.size == 0:
        raise ValueError("CSV appears empty after parsing.")
    return names, data, delim

def _pick_columns_np(names: Optional[List[str]], data: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Select time (optional) and ECG (required) columns from a 2D float array.

    Heuristics:
      - If names are present: prefer column whose name contains 'time' (case-insensitive) as time.
        Prefer column whose name contains 'ecg' as ECG.
      - Otherwise (or if not found):
           • Consider the 1st column a time candidate if it's strictly increasing (monotonic) and spans > N/fs.
           • Choose ECG as the non-time column with highest variance.
    """
    n_rows, n_cols = data.shape
    if n_cols == 0:
        raise ValueError("No columns found in ECG CSV.")

    time_idx: Optional[int] = None
    ecg_idx: Optional[int]  = None

    if names is not None and len(names) == n_cols:
        names_lower = [str(c).lower() for c in names]
        # time by header
        for i, nm in enumerate(names_lower):
            if "time" in nm:
                time_idx = i
                break
        # ecg by header
        for i, nm in enumerate(names_lower):
            if "ecg" in nm:
                ecg_idx = i
                break

    # If no header-based time, try monotonic heuristic on col0
    if time_idx is None and n_cols >= 1:
        t0 = data[:, 0]
        # monotonic increasing and not constant
        if np.all(np.isfinite(t0)) and np.all(np.diff(t0) >= 0) and (t0[-1] > t0[0]):
            time_idx = 0

    # If no header-based ECG, choose the non-time column with highest variance
    if ecg_idx is None:
        cand = [i for i in range(n_cols) if i != (time_idx if time_idx is not None else -1)]
        if not cand:
            cand = list(range(n_cols))
        vari = [np.nanvar(data[:, i]) for i in cand]
        ecg_idx = cand[int(np.argmax(vari))]

    # Extract arrays
    t = data[:, time_idx].astype(float) if time_idx is not None else None
    x = data[:, ecg_idx].astype(float)
    return t, x

# ---------------------------------------------------------------

def _resample_to_fs(t: Optional[np.ndarray], x: np.ndarray, fs: float) -> np.ndarray:
    """
    If timestamps present, resample to uniform fs via linear interpolation.
    Else, assume current sampling is already uniform; if not, we’ll still
    treat as uniform at fs (best effort).
    """
    if t is None:
        return x.astype(np.float32)

    # sanitize time
    t = np.asarray(t, dtype=float)
    # ensure monotonic
    order = np.argsort(t)
    t = t[order]; x = x[order]
    t0, t1 = t[0], t[-1]
    if t1 <= t0:
        raise ValueError("Invalid timestamps in ECG CSV.")
    n = int(np.floor((t1 - t0) * fs)) + 1
    t_new = t0 + np.arange(n) / fs
    x_new = np.interp(t_new, t, x)
    return x_new.astype(np.float32)

def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    b, a = butter(order, [lo/(fs/2.0), hi/(fs/2.0)], btype="band")
    # guard for very short arrays
    return filtfilt(b, a, x) if len(x) > (order*3) else x

def _r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    # simple QRS proxy on squared derivative + distance constraint
    dx2 = np.diff(ecg, prepend=ecg[0])**2
    thr = dx2.mean() + 2.0*dx2.std()
    cand, _ = find_peaks(dx2, distance=int(0.25*fs))
    ridx = cand[dx2[cand] > thr]
    return ridx

def _slice_windows(ecg: np.ndarray, ridx: np.ndarray, fs: float,
                   pre: float, post: float,
                   rr_minmax: Tuple[float,float]) -> List[np.ndarray]:
    n_pre, n_post = int(pre*fs), int(post*fs)
    wins = []
    for i in range(len(ridx)-1):
        rr = (ridx[i+1]-ridx[i])/fs
        if not (rr_minmax[0] <= rr <= rr_minmax[1]):
            continue
        s = ridx[i] - n_pre
        e = ridx[i] + n_post
        if s < 0 or e > len(ecg):
            continue
        w = ecg[s:e]
        if np.all(np.isfinite(w)):
            wins.append(w.astype(np.float32))
    return wins  # each length L = int((pre+post)*fs) = 90 at fs=100

def _stft_tile(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-beat STFT log-magnitude spectrogram with your training params.
    Returns (S[F,T], f, t)
    """
    # per-beat z-score (stabilize across beats)
    x = (x - x.mean()) / (x.std() + 1e-8)
    f, t, Z = stft(
        x, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
        detrend=False, return_onesided=True, boundary=None, padded=False
    )
    # crop frequency band
    imax = np.searchsorted(f, FMAX)
    f = f[:imax]; Z = Z[:imax, :]
    S = 20.0 * np.log10(np.abs(Z) + 1e-12).astype(np.float32)  # log-magnitude
    # optional per-beat normalization of the spectrogram itself (comment out if not used in training)
    m = S.mean(); s = S.std() + 1e-8
    S = (S - m) / s
    return S, f, t

def _resize_2d_linear(M: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Resize a 2D array to (H,W) by separable linear interpolation (no extra deps).
    """
    h0, w0 = M.shape
    if h0 == H and w0 == W:
        return M.astype(np.float32)

    y = np.linspace(0, h0 - 1, H, dtype=np.float32)
    x = np.linspace(0, w0 - 1, W, dtype=np.float32)

    # interpolate rows -> (H, w0)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, h0 - 1)
    wy = (y - y0).reshape(-1, 1)
    M_y = (1 - wy) * M[y0, :] + wy * M[y1, :]

    # interpolate cols -> (H, W)
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, w0 - 1)
    wx = (x - x0).reshape(1, -1)
    M_xy = (1 - wx) * M_y[:, x0] + wx * M_y[:, x1]
    return M_xy.astype(np.float32)

def ecg_csv_to_stft_tiles(
    file_obj: Union[str, bytes, BinaryIO],
    fs_target: float = FS,
    win_pre: float = WIN_PRE,
    win_post: float = WIN_POST,
) -> np.ndarray:
    """
    Read an uploaded ECG CSV (Streamlit file/path/bytes), preprocess like in your
    vitals pipeline, and return an array of STFT spectrogram tiles for the model.

    Returns:
      X: np.ndarray with shape (N, 58, 17, 1), dtype=float32
         (exactly what your vitals model expects)
    """
    names, data, _ = _load_ecg_csv_np(file_obj)        # (names or None, data: [N, C])
    t, ecg = _pick_columns_np(names, data)             # (time or None, ecg series)

    # resample to target fs if timestamps present
    ecg = _resample_to_fs(t, ecg, fs_target)
    # bandpass 0.5–40 Hz
    ecg = _bandpass(np.nan_to_num(ecg, nan=0.0), fs_target, BP_LOHI[0], BP_LOHI[1])

    # detect R-peaks
    ridx = _r_peaks(ecg, fs_target)
    if len(ridx) < 3:
        raise RuntimeError("ECG too short or poor quality: insufficient R-peaks detected.")

    # slice windows around each R with RR sanity check
    wins = _slice_windows(ecg, ridx, fs_target, win_pre, win_post, RR_MINMAX)
    if not wins:
        raise RuntimeError("No clean ECG windows extracted after filtering.")

    # STFT tiles -> resize to (58,17) -> add channel -> stack
    tiles = []
    for w in wins:
        S, _, _ = _stft_tile(w, fs_target)                # (F,T)
        S = _resize_2d_linear(S, OUT_FBINS, OUT_TBINS)    # (58,17)
        tiles.append(S[..., None])                         # (58,17,1)

    X = np.stack(tiles, axis=0).astype(np.float32)        # (N,58,17,1)
    return X