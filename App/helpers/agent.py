# App/helpers/agent.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List

from helpers.rag_index import load_or_build_index, retrieve

# ---------------------------
# Module-global RAG singleton
# ---------------------------
_RAG = None  # lazily initialized once and reused

def _ensure_rag(pdf_dir: str, cache_path: str):
    global _RAG
    if _RAG is None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        _RAG = load_or_build_index(pdf_dir, cache_path)
    return _RAG

# ---------------------------
# Simple cross-checks (no diagnosis)
# ---------------------------
def _to_float(s) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")

def cross_check(row: Dict[str, str]) -> List[str]:
    flags: List[str] = []
    MAP   = _to_float(row.get("MAP_mmHg", "nan"))
    SpO2  = _to_float(row.get("SpO2_pct", "nan"))
    ETCO2 = _to_float(row.get("ETCO2_mmHg", "nan"))

    if MAP == MAP and MAP < 65:    flags.append(f"Low MAP {MAP:.1f} mmHg (hypotension risk).")
    if MAP == MAP and MAP > 110:   flags.append(f"High MAP {MAP:.1f} mmHg.")
    if SpO2 == SpO2 and SpO2 < 92: flags.append(f"Low SpO₂ {SpO2:.1f}%.")
    if ETCO2 == ETCO2 and ETCO2 < 30: flags.append(f"Low ETCO₂ {ETCO2:.1f} mmHg.")
    if ETCO2 == ETCO2 and ETCO2 > 45: flags.append(f"High ETCO₂ {ETCO2:.1f} mmHg.")

    phase = (row.get("Phase") or "").lower()
    insts = [s.strip().lower() for s in (row.get("Instrument") or "").split(",") if s.strip()]
    targs = [s.strip().lower() for s in (row.get("Target") or "").split(",") if s.strip()]

    if "clipping" in phase and "clipper" not in insts:
        flags.append("Phase suggests clipping but clipper not detected.")
    if "dissection" in phase and not any(t in insts for t in ["hook", "scissors", "grasper"]):
        flags.append("Dissection phase without typical dissecting tools detected.")
    if "cystic_duct" in targs and "clipping" in phase and "clipper" not in insts:
        flags.append("Cystic duct target during clipping but clipper missing.")

    return flags

def _query_from_row(row: Dict[str, str], flags: List[str]) -> str:
    base = (
        f"Cholecystectomy context. "
        f"Phase: {row.get('Phase','')}. "
        f"Tools: {row.get('Instrument','')}. "
        f"Targets: {row.get('Target','')}. "
        f"Vitals: MAP={row.get('MAP_mmHg','')}, SpO2={row.get('SpO2_pct','')}, ETCO2={row.get('ETCO2_mmHg','')}."
    )
    if not flags:
        return base + " Provide concise best-practice reminders for this phase in 1–2 sentences."
    return base + " Issues: " + "; ".join(flags) + " Provide concise, actionable checks with brief rationale."

# ---------------------------
# Groq LLM backend only
# ---------------------------
def _llm_groq(prompt: str, model: str) -> str:
    """
    Fast hosted LLM via Groq. Requires env var GROQ_API_KEY.
    """
    api_key = "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" #"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"
    if not api_key:
        return "Insight: (GROQ_API_KEY not set; cannot generate LLM insight)."
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a surgical assistant. Be concise. Safety-oriented, non-diagnostic guidance only. Strictly do not give two insights by using the word 'or' in between and be very clear"},
                {"role":"user","content":prompt[:6000]},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Insight: (Groq call failed: {e})"

# ---------------------------
# Agent (depends on model_runner outputs + rag_index)
# ---------------------------
@dataclass
class AgentConfig:
    pdf_dir: str = "App/LLM/Data_Files"
    cache_path: str = "Preprocessed_Files/llm_rag_index"
    groq_model: str = "llama-3.1-8b-instant"
    top_k: int = 4

class SurgicalLLMAgent:
    """
    Accepts a single row from run_per_frame_predictions(...) and returns INSIGHT TEXT.
    Database is created once (cached) via rag_index.load_or_build_index and reused.
    """
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.rag = _ensure_rag(cfg.pdf_dir, cfg.cache_path)

    def analyze_frame(self, frame_pred: Dict[str, str]) -> str:
        # 1) Rules
        flags = cross_check(frame_pred)

        # 2) Retrieve top-k guideline snippets (NumPy cosine – no FAISS)
        query = _query_from_row(frame_pred, flags)
        docs = retrieve(self.rag, query, k=self.cfg.top_k)

        # 3) Build concise, cited prompt
        cites = []
        
        for i, (txt, (fname, idx), score) in enumerate(docs, 1):
            cites.append(f"[{i}] ({fname} #{idx}) {txt[:450]}")
            
        prompt = (
            "Using model outputs and retrieved cholecystectomy guidelines, produce a short and crisp, real-time operative insight which a surgeon can quickly interpret in realtime.It should not exceed 35 words and the insight should be detailed only when you detect any anomaly otherwise it should be very short in one line "
            "(<=3 sentences). Cite snippets as [1],[2]. Do not give diagnosis or treatment.\n\n"
            f"Model outputs:\n{frame_pred}\n\n"
            "Guideline snippets:\n" + "\n\n".join(cites)
        )

        # 4) LLM (Groq)
        return _llm_groq(prompt, self.cfg.groq_model)

    def analyze_rows(self, rows: List[Dict[str, str]]) -> List[str]:
        return [self.analyze_frame(r) for r in rows]
