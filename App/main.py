# App/main.py
from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import threading
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# --------------------- env & crash traces ---------------------
os.environ.setdefault("PANDAS_IGNORE_PYARROW", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import faulthandler
faulthandler.enable(all_threads=True)

def _dump_stacks(signum, frame):
    faulthandler.dump_traceback(all_threads=True)

if threading.current_thread() is threading.main_thread():
    try:
        signal.signal(signal.SIGUSR1, _dump_stacks)
    except Exception:
        pass

mp.set_start_method("spawn", force=True)

# --------------------- utilities ---------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

# report helper
try:
    from App.helpers.report import save_report
except ModuleNotFoundError:
    from helpers.report import save_report

# ---------- background helpers (robust) ----------
def _find_background_image() -> str | None:
    img_dir = ROOT / "App" / "App_Background_Image"
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    for pat in patterns:
        files = sorted(img_dir.glob(pat))
        if files:
            return str(files[0].resolve())
    return None

def _apply_background_abs(abs_img_path: str | None):
    css_base = """
    <style>
    .block-container { background: transparent !important; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
    [data-testid="stSidebar"] { background: rgba(0,0,0,0) !important; }

    .glass {
        background: rgba(255,255,255,0.72);
        -webkit-backdrop-filter: blur(8px);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 16px 20px;
        border: 1px solid rgba(255,255,255,0.35);
    }

    .stSelectbox > div, .stFileUploader > div {
        background: rgba(0,0,0,0.06);
        border-radius: 10px;
    }
    </style>
    """
    st.markdown(css_base, unsafe_allow_html=True)

    if not abs_img_path or not os.path.exists(abs_img_path):
        st.warning("Background image not found under `App/App_Background_Image/`. Put a .png/.jpg there.")
        return

    with open(abs_img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    css_bg = f"""
    <style>
    body {{
        background: url("data:image;base64,{b64}") center center / cover no-repeat fixed !important;
    }}
    [data-testid="stAppViewContainer"] {{
        background: url("data:image;base64,{b64}") center center / cover no-repeat fixed !important;
    }}
    </style>
    """
    st.markdown(css_bg, unsafe_allow_html=True)

# --- TTS: persistent controller UI + lightweight enqueue ----
def _render_tts_controller_once():
    """
    Renders ONE controller (no key arg) that:
      - maintains a localStorage queue
      - provides Enable/Disable toggle
      - plays queue sequentially when enabled
    Safe to call each run; it just redefines the same functions/UI.
    """
    components.html("""
<div class="glass" style="padding:8px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <button id="tts-toggle" style="padding:6px 10px;border-radius:8px;border:1px solid #ddd;background:#000;color:#fff;cursor:pointer;">…</button>
    <small id="tts-status" style="color:#555;">Browser audio; click once to allow.</small>
  </div>
</div>
<script>
(function(){
  try {
    const synth = window.speechSynthesis;

    // one-time init
    if (!localStorage.getItem('tts_queue'))   localStorage.setItem('tts_queue','[]');
    if (!localStorage.getItem('tts_rate'))    localStorage.setItem('tts_rate','1.0');
    if (!localStorage.getItem('tts_vol'))     localStorage.setItem('tts_vol','1.0');
    if (!localStorage.getItem('tts_enabled')) localStorage.setItem('tts_enabled','0');

    function dequeueAndSpeak(){
      if (localStorage.getItem('tts_enabled') !== '1') return;
      if (synth.speaking || synth.pending) return;

      const q = JSON.parse(localStorage.getItem('tts_queue')||'[]');
      if (!q.length) return;

      const txt = q.shift();
      localStorage.setItem('tts_queue', JSON.stringify(q));

      const u = new SpeechSynthesisUtterance(txt);
      const vs = synth.getVoices();
      const v  = vs.find(v => /en/i.test(v.lang) || /en/i.test(v.name));
      if (v) u.voice = v;
      u.rate   = parseFloat(localStorage.getItem('tts_rate')||'1.0');
      u.volume = parseFloat(localStorage.getItem('tts_vol') || '1.0');
      u.pitch  = 1.0;
      u.onend  = () => setTimeout(dequeueAndSpeak, 120);
      synth.speak(u);
    }
    setInterval(dequeueAndSpeak, 250);

    // UI toggle
    const btn   = document.getElementById('tts-toggle');
    const label = document.getElementById('tts-status');
    function render(){
      const on = localStorage.getItem('tts_enabled') === '1';
      btn.textContent = on ? '🔇 Disable voice' : '🔊 Enable voice';
      label.textContent = on ? 'Voice is ON' : 'Voice is OFF (click to enable)';
    }
    btn.onclick = function(){
      const on = localStorage.getItem('tts_enabled') === '1';
      if (on) {
        localStorage.setItem('tts_enabled','0');
        try { synth.cancel(); } catch(e) {}
        localStorage.setItem('tts_queue','[]');
      } else {
        localStorage.setItem('tts_enabled','1');
        try {
          const u = new SpeechSynthesisUtterance('Voice enabled');
          synth.cancel(); synth.speak(u);
        } catch(e) {}
      }
      render();
    };
    render();
  } catch(e) { console.warn(e); }
})();
</script>
    """, height=60)

def _tts_enqueue(text: str):
    """Push text to the queue without re-creating the controller iframe."""
    safe = (text or "").replace("\\","\\\\").replace("`","\\`")
    components.html(f"""
<script>
try {{
  const q = JSON.parse(localStorage.getItem('tts_queue')||'[]');
  q.push(`{safe}`);
  localStorage.setItem('tts_queue', JSON.stringify(q));
}} catch(e) {{}}
</script>
    """, height=0)

# ---------- path helpers ----------
def _resolve_pdf_dir(pdf_dir_input: str) -> Path:
    app_dir = Path(__file__).resolve().parent
    proj_root = app_dir.parent
    cand: List[Path] = []
    if pdf_dir_input:
        p = Path(pdf_dir_input)
        if p.is_absolute(): cand.append(p)
        else:
            cand.append(app_dir / p)
            cand.append(proj_root / p)
    cand.append(app_dir / "LLM" / "Data_Files")
    cand.append(proj_root / "App" / "LLM" / "Data_Files")
    for c in cand:
        if c.exists() and c.is_dir() and any(c.glob("*.pdf")):
            return c.resolve()
    tried = "\n".join(str(c) for c in cand)
    raise FileNotFoundError(f"No PDFs found. Tried:\n{tried}")

def _resolve_models_dir(models_dir_input: str) -> Path:
    app_dir = Path(__file__).resolve().parent
    proj_root = app_dir.parent
    cand: List[Path] = []
    if models_dir_input:
        p = Path(models_dir_input)
        if p.is_absolute(): cand.append(p)
        else:
            cand.append(app_dir / p)
            cand.append(proj_root / p)
    else:
        cand.append(app_dir / "Saved_Models")
        cand.append(proj_root / "Saved_Models")
    for c in cand:
        if c.exists() and c.is_dir():
            return c.resolve()
    tried = "\n".join(str(c) for c in cand)
    raise FileNotFoundError(
        f"Models directory not found. Tried:\n{tried}\n"
        "Set 'Models directory' in the UI to a valid folder."
    )

def _auto_discover_models(models_dir: str) -> Dict[str, str]:
    import glob, os
    want = {
        "action":      ("action",),
        "phase":       ("phase",),
        "target":      ("target",),
        "instrument":  ("instrument", "tool"),
        "vitals":      ("vitals", "vital"),
    }
    found: Dict[str, str] = {}
    candidates = glob.glob(os.path.join(models_dir, "*version_1.keras"))
    names = [os.path.basename(p).lower() for p in candidates]
    if not candidates:
        st.warning(f"No *version_1.keras files in: {models_dir}")
    for key, aliases in want.items():
        for p, name in zip(candidates, names):
            if any(a in name for a in aliases):
                found[key] = p
                break
    missing = [k for k in want.keys() if k not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing model files for: {', '.join(missing)}. "
            f"Place *version_1.keras files in {models_dir} with names containing these keywords "
            f"(accepting synonyms): action | phase | target | instrument(tool) | vitals(vital)."
        )
    return found

def _read_ecg_csv_uploadedfile(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        from App.input_processing.signal_to_image_processing import ecg_csv_to_stft_tiles
    except ModuleNotFoundError:
        from input_processing.signal_to_image_processing import ecg_csv_to_stft_tiles
    try:
        uploaded_file.seek(0)
        return ecg_csv_to_stft_tiles(uploaded_file)
    except Exception:
        tmp = "Preprocessed_Files/_tmp_ecg.csv"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return ecg_csv_to_stft_tiles(tmp)

# --------------------- page setup ---------------------
st.set_page_config(page_title="Surgical AI Agent", layout="wide")
_apply_background_abs(_find_background_image())

st.markdown("<h1 style='color:#ff2d2d; text-shadow:0 1px 2px rgba(0,0,0,.15)'>Surgical AI Agent</h1>", unsafe_allow_html=True)
st.caption("Per-frame multi-model predictions + RAG-assisted LLM insights for laparoscopic cholecystectomy.")

# --------------------- session state ---------------------
if "rows" not in st.session_state:        st.session_state.rows = None
if "insights" not in st.session_state:    st.session_state.insights = None
if "frame_count" not in st.session_state: st.session_state.frame_count = 0
if "report_paths" not in st.session_state: st.session_state.report_paths = None

# --------------------- layout ---------------------
left, mid, right = st.columns([0.70, 0.02, 0.30], gap="large")

# ===== LEFT pane =====
with left:
    st.markdown("### Upload inputs")

    c_vid, c_ecg = st.columns([1,1], gap="large")
    with c_vid:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Video")
        video_file = st.file_uploader("Upload operative video", type=["mp4","mov","avi","mkv","mpeg4"])
        every_n_seconds = st.slider("Frame sample interval (s)", 0.2, 3.0, 1.0, 0.2)
        size_options = [(192,192),(224,224),(256,256)]
        target_size = st.selectbox("Frame size (W,H)", options=size_options, index=1,
                                   format_func=lambda s: f"{s[0]}×{s[1]}")
        crop_square = st.checkbox("Center crop square", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_ecg:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("ECG (optional)")
        ecg_file = st.file_uploader("Upload ECG CSV (optional)", type=["csv"])
        st.caption("If provided, vitals are predicted per frame; otherwise shown as NaN.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Settings")
    c_models, c_llm, c_vitals = st.columns([1,1,1], gap="large")

    with c_models:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Models")
        models_dir = st.text_input("Models directory", value="Saved_Models")
        auto_discover = st.checkbox("Auto-detect model files by name", value=True)
        manual_paths: Dict[str,str] | None = None
        if not auto_discover:
            manual_paths = {
                "instrument": st.text_input("Instrument model", ""),
                "action":     st.text_input("Action model", ""),
                "phase":      st.text_input("Phase model", ""),
                "target":     st.text_input("Target model", ""),
                "vitals":     st.text_input("Vitals model", ""),
            }
        st.markdown("</div>", unsafe_allow_html=True)

    with c_llm:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("LLM Backend")
        backend = st.selectbox("Backend", ["groq", "llamacpp"], index=0)
        groq_model = st.selectbox("Groq model", ["llama-3.1-8b-instant","mixtral-8x7b-instruct"])
        gguf_path  = st.text_input("GGUF path (for llama.cpp)", value="")
        top_k = st.slider("Guideline snippets (top-k)", 1, 8, 4)
        #st.caption("Set GROQ_API_KEY for Groq; provide a real .gguf for llama.cpp.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_vitals:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Vitals scaling")
        use_denorm = st.checkbox("Convert outputs to real units with DB stats", value=False)
        y_mean = y_std = None
        if use_denorm:
            y_mean_str = st.text_input("y_mean [MAP, SpO2, ETCO2]", "92.3,97.8,33.1")
            y_std_str  = st.text_input("y_std  [MAP, SpO2, ETCO2]", "14.8,1.7,3.2")
            try:
                y_mean = np.array([float(x.strip()) for x in y_mean_str.split(",")], dtype=np.float32)
                y_std  = np.array([float(x.strip()) for x in y_std_str.split(",")], dtype=np.float32)
                if y_mean.shape != (3,) or y_std.shape != (3,):
                    raise ValueError
            except Exception:
                st.error("Provide 3 comma-separated floats for y_mean and y_std.")
                y_mean = None; y_std = None
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Voice controller (same place as before) ----
    _render_tts_controller_once()

    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ===== MIDDLE separator =====
with mid:
    st.markdown(
        """
        <div style="
            border-left: 2px solid rgba(255,255,255,0.25);
            height: calc(100vh - 140px);
            margin-top: 8px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== RIGHT pane (persistent) =====
with right:
    st.markdown("### Surgical Bot")
    if st.session_state.rows is None or st.session_state.insights is None:
        st.info("Upload inputs on the left and click **Run analysis**. Insights will appear here.")
    else:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        max_idx = max(0, st.session_state.frame_count - 1)
        if max_idx >= 1:
            idx = st.slider("Frame", 0, max_idx, 0, 1, key="frame_slider")
        else:
            idx = 0
            st.caption("Frame 1/1")

        cA, cB = st.columns([1,1], gap="large")
        with cA:
            st.subheader("Predictions (per frame)")
            st.json(st.session_state.rows[idx], expanded=False)
        with cB:
            st.subheader("LLM Insight")
            st.write(st.session_state.insights[idx])

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Persistent Download block (if already generated earlier) ----
    if st.session_state.get("report_paths"):
        paths = st.session_state.report_paths
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Download Report")
        st.write("CSV, HTML (pretty), and optional PDF if `weasyprint` is installed.")

        if os.path.exists(paths.get("html","")):
            with open(paths["html"], "rb") as f:
                st.download_button("⬇️ Download HTML report", f, file_name="report.html",
                                   mime="text/html", use_container_width=True, key="dl_html_persist")
        if "pdf" in paths and os.path.exists(paths["pdf"]):
            with open(paths["pdf"], "rb") as f:
                st.download_button("⬇️ Download PDF report", f, file_name="report.pdf",
                                   mime="application/pdf", use_container_width=True, key="dl_pdf_persist")
        if os.path.exists(paths.get("csv","")):
            with open(paths["csv"], "rb") as f:
                st.download_button("⬇️ Download per-frame CSV", f, file_name="preds.csv",
                                   mime="text/csv", use_container_width=True, key="dl_csv_persist")
        st.caption(f"Saved to: `{Path(paths['html']).parent}`")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RUN after button click ----------------
if run_btn:
    try:
        # heavy imports at run time
        try:
            from App.input_processing.video_to_image_processing import extract_frames_rgb
            from App.helpers.model_runner import run_per_frame_predictions
            from App.helpers.agent import SurgicalLLMAgent, AgentConfig
        except ModuleNotFoundError:
            from input_processing.video_to_image_processing import extract_frames_rgb
            from helpers.model_runner import run_per_frame_predictions
            from helpers.agent import SurgicalLLMAgent, AgentConfig

        # --- models ---
        with st.status("Finding models...", expanded=False) as s:
            if auto_discover:
                models_dir_abs = _resolve_models_dir(models_dir)
                st.caption(f"Searching models in: `{models_dir_abs}`")
                paths = _auto_discover_models(str(models_dir_abs))
            else:
                paths = manual_paths or {}
                if any(not v for v in paths.values()) or set(paths.keys()) != {"instrument","action","phase","target","vitals"}:
                    raise FileNotFoundError("Provide all five model paths in manual mode.")
            s.update(label="Models found.", state="complete")

        # --- frames ---
        with st.status("Extracting frames...", expanded=False) as s:
            if video_file is None:
                raise ValueError("Please upload a video file.")
            frames = extract_frames_rgb(
                video=video_file,  # UploadedFile is file-like and supported
                every_n_seconds=every_n_seconds,
                target_size=target_size,
                crop_square=crop_square,
            )
            if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3 or frames.size == 0:
                raise RuntimeError("No frames extracted. Check the video.")
            s.update(label=f"Frames ready: {len(frames)}", state="complete")

        # --- ECG -> STFT ---
        stft_tiles = None
        if ecg_file is not None:
            with st.status("Processing ECG...", expanded=False) as s:
                stft_4d = _read_ecg_csv_uploadedfile(ecg_file)   # -> (N,58,17,1)
                stft_tiles = stft_4d[..., 0]                     # -> (N,58,17)
                s.update(label=f"ECG processed. STFT shape: {stft_tiles.shape}", state="complete")

        # --- run CV/vitals models ---
        with st.status("Running models...", expanded=False) as s:
            rows = run_per_frame_predictions(
                frames=frames,
                stft_array=stft_tiles,
                paths=paths,
                y_mean=y_mean,
                y_std=y_std,
                max_workers=8,
            )
            s.update(label="Model predictions complete.", state="complete")

        # --- LLM insights (stream, per frame) ---
        with st.status("Generating LLM insights...", expanded=False) as s:
            pdf_dir_abs = _resolve_pdf_dir("App/LLM/Data_Files")
            cache_abs   = (Path(__file__).resolve().parent.parent / "Preprocessed_Files" / "llm_rag_index").resolve()
            st.caption(f"Using guideline PDFs from: `{pdf_dir_abs}`")

            agent = SurgicalLLMAgent(AgentConfig(
                pdf_dir=str(pdf_dir_abs),
                cache_path=str(cache_abs),
                groq_model=groq_model,
                top_k=top_k,
            ))

            total = len(rows)

            # Live counters/outputs (no extra sliders here; keeps right-pane static)
            with right:
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                live_count_ph = st.empty()      # shows "Frame i/N" live
                pred_col, insight_col = st.columns([1, 1], gap="large")
                pred_col.subheader("Predictions (per frame)")
                insight_col.subheader("LLM Insight")
                pred_ph = pred_col.empty()
                insight_ph = insight_col.empty()
                st.markdown("</div>", unsafe_allow_html=True)

            insights: list[str] = []
            prog = st.progress(0.0, text="Analyzing frames...")

            for i, row in enumerate(rows):
                try:
                    insight = agent.analyze_frame(row)
                except Exception as e:
                    insight = f"(Insight error: {e})"

                insights.append(insight)

                # Live UI updates
                live_count_ph.markdown(f"**Frame {i+1}/{total}**")
                pred_ph.json(row, expanded=False)
                insight_ph.write(insight)

                # Enqueue this frame's insight (sequential playback via controller)
                _tts_enqueue(insight)

                prog.progress((i + 1) / total, text=f"Analyzed {i + 1}/{total} frames")

            s.update(label="Insights ready.", state="complete")

        # Save to session for persistent right-pane browsing
        st.session_state.rows = rows
        st.session_state.insights = insights
        st.session_state.frame_count = len(rows)

        # ---- Build downloadable report (persist paths in session) ---
        session_dir = Path("Reports") / f"session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        paths = save_report(
            rows=rows,
            insights=insights,
            output_dir=str(session_dir),
            video_name=getattr(video_file, "name", None) if video_file is not None else None,
        )
        st.session_state.report_paths = paths  # persists buttons even after click

        # ---- Also render the Download block immediately (no rerun) ----
        with right:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.subheader("Download Report")
            st.write("CSV, HTML (pretty), and optional PDF if `weasyprint` is installed.")

            if os.path.exists(paths.get("html","")):
                with open(paths["html"], "rb") as f:
                    st.download_button("⬇️ Download HTML report", f, file_name="report.html",
                                mime="text/html", use_container_width=True, key="dl_html_now")
            if "pdf" in paths and os.path.exists(paths["pdf"]):
                with open(paths["pdf"], "rb") as f:
                    st.download_button("⬇️ Download PDF report", f, file_name="report.pdf",
                                    mime="application/pdf", use_container_width=True, key="dl_pdf_now")
            if os.path.exists(paths.get("csv","")):
                with open(paths["csv"], "rb") as f:
                    st.download_button("⬇️ Download per-frame CSV", f, file_name="preds.csv",
                                    mime="text/csv", use_container_width=True, key="dl_csv_now")
            st.caption(f"Saved to: `{session_dir}`")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.exception(e)
