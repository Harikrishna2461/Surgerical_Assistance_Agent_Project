# App/helpers/report.py
from __future__ import annotations
import os, io, json, base64, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import math
# 4) Optional PDF (if weasyprint exists + system libs available)
pdf_path = ""
try:
    from weasyprint import HTML  # lazy import; only tried when we actually build PDF
    pdf_path = str(Path("App/Surgery_Report") / "report.pdf")
    HTML(string=html).write_pdf(pdf_path)
except Exception:
    # No system libs or weasyprint not installed; skip PDF silently
    pdf_path = ""


# ---------- tiny, safe number formatter ----------
def _fmt2(x: object) -> str:
    """Format as '%.2f' unless None/NaN -> em dash."""
    try:
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:.2f}"
    except Exception:
        return "—"

# ---------- helpers ----------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def summarize_rows(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    N = len(rows)
    # Tallies
    inst_counts: Dict[str, int] = {}
    phase_counts: Dict[str, int] = {}
    targ_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}

    MAP, SpO2, ETCO2 = [], [], []

    flags_low_map = flags_low_spo2 = flags_low_etco2 = 0
    flags_high_map = flags_high_etco2 = 0

    for r in rows:
        # instruments: comma separated
        for tok in [s.strip() for s in (r.get("Instrument") or "").split(",") if s.strip()]:
            inst_counts[tok] = inst_counts.get(tok, 0) + 1
        # phase
        ph = (r.get("Phase") or "").strip()
        if ph:
            phase_counts[ph] = phase_counts.get(ph, 0) + 1
        # targets
        for tok in [s.strip() for s in (r.get("Target") or "").split(",") if s.strip()]:
            targ_counts[tok] = targ_counts.get(tok, 0) + 1
        # action
        act = (r.get("Surgical_Action") or "").strip()
        if act:
            action_counts[act] = action_counts.get(act, 0) + 1

        m = _safe_float(r.get("MAP_mmHg"))
        s = _safe_float(r.get("SpO2_pct"))
        e = _safe_float(r.get("ETCO2_mmHg"))
        if np.isfinite(m): MAP.append(m)
        if np.isfinite(s): SpO2.append(s)
        if np.isfinite(e): ETCO2.append(e)

        if np.isfinite(m) and m < 65: flags_low_map += 1
        if np.isfinite(m) and m > 110: flags_high_map += 1
        if np.isfinite(s) and s < 92: flags_low_spo2 += 1
        if np.isfinite(e) and e < 30: flags_low_etco2 += 1
        if np.isfinite(e) and e > 45: flags_high_etco2 += 1

    def pct(x): 
        return (100.0 * x / max(1, N))

    summary = {
        "n_frames": N,
        "inst_counts": sorted(inst_counts.items(), key=lambda kv: -kv[1])[:10],
        "phase_counts": sorted(phase_counts.items(), key=lambda kv: -kv[1]),
        "targ_counts": sorted(targ_counts.items(), key=lambda kv: -kv[1])[:10],
        "action_counts": sorted(action_counts.items(), key=lambda kv: -kv[1])[:10],
        "vitals": {
            "MAP_mean": float(np.nanmean(MAP)) if MAP else None,
            "MAP_std": float(np.nanstd(MAP)) if MAP else None,
            "SpO2_mean": float(np.nanmean(SpO2)) if SpO2 else None,
            "SpO2_std": float(np.nanstd(SpO2)) if SpO2 else None,
            "ETCO2_mean": float(np.nanmean(ETCO2)) if ETCO2 else None,
            "ETCO2_std": float(np.nanstd(ETCO2)) if ETCO2 else None,
        },
        "flags": {
            "low_MAP_frames": flags_low_map,
            "high_MAP_frames": flags_high_map,
            "low_SpO2_frames": flags_low_spo2,
            "low_ETCO2_frames": flags_low_etco2,
            "high_ETCO2_frames": flags_high_etco2,
            "low_MAP_pct": pct(flags_low_map),
            "high_MAP_pct": pct(flags_high_map),
            "low_SpO2_pct": pct(flags_low_spo2),
            "low_ETCO2_pct": pct(flags_low_etco2),
            "high_ETCO2_pct": pct(flags_high_etco2),
        }
    }
    return summary

def _escape(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _inline_css() -> str:
    return """
    <style>
      body { font-family: Inter, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color:#131722; }
      .container { max-width: 1080px; margin: 20px auto; }
      .title { color:#c21807; margin-bottom: 6px; }
      .subtle { color:#555; }
      .card { background:#fff; border:1px solid #eee; border-radius:12px; padding:16px 18px; margin:12px 0; box-shadow:0 2px 10px rgba(0,0,0,.04); }
      .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
      .kpi { background:#fafafa; border:1px solid #f0f0f0; border-radius:10px; padding:12px; }
      h2 { margin: 6px 0 8px; font-size: 18px; }
      h3 { margin: 4px 0 6px; font-size: 15px; color:#333; }
      table { width:100%; border-collapse:collapse; }
      th, td { border-bottom:1px solid #eee; padding:8px 10px; text-align:left; font-size: 13px; }
      th { background:#fafafa; }
      .pill { display:inline-block; background:#eef4ff; color:#1b4fd9; padding:3px 8px; border-radius:999px; margin:2px 4px 2px 0; font-size:12px; }
      .warn { color:#b71c1c; font-weight:600; }
      .muted { color:#777; }
      .foot { margin-top:10px; color:#777; font-size:12px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    </style>
    """

def _list_pills(items) -> str:
    return "".join(f"<span class='pill'>{_escape(k)}&nbsp;({_escape(str(v))})</span>" for k,v in items)

def _render_summary_html(summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    vit = summary["vitals"]
    flags = summary["flags"]

    return f"""
    <div class="card">
      <h2>Case Summary</h2>
      <div class="grid">
        <div class="kpi">
          <h3>Frames</h3>
          <div class="mono">{summary['n_frames']}</div>
        </div>
        <div class="kpi">
          <h3>Video</h3>
          <div class="mono">{_escape(meta.get('video_name', '—'))}</div>
        </div>
      </div>

      <div class="grid">
        <div class="kpi">
          <h3>Vitals (mean ± sd)</h3>
          <div class="mono">MAP: {_fmt2(vit.get('MAP_mean'))} ± {_fmt2(vit.get('MAP_std'))} mmHg</div>
          <div class="mono">SpO₂: {_fmt2(vit.get('SpO2_mean'))} ± {_fmt2(vit.get('SpO2_std'))} %</div>
          <div class="mono">ETCO₂: {_fmt2(vit.get('ETCO2_mean'))} ± {_fmt2(vit.get('ETCO2_std'))} mmHg</div>
        </div>
        <div class="kpi">
          <h3>Safety Flags</h3>
          <div class="mono">Low MAP: {flags['low_MAP_frames']} ({flags['low_MAP_pct']:.1f}%)</div>
          <div class="mono">High MAP: {flags['high_MAP_frames']} ({flags['high_MAP_pct']:.1f}%)</div>
          <div class="mono">Low SpO₂: {flags['low_SpO2_frames']} ({flags['low_SpO2_pct']:.1f}%)</div>
          <div class="mono">Low ETCO₂: {flags['low_ETCO2_frames']} ({flags['low_ETCO2_pct']:.1f}%)</div>
          <div class="mono">High ETCO₂: {flags['high_ETCO2_frames']} ({flags['high_ETCO2_pct']:.1f}%)</div>
        </div>
      </div>

      <h3>Most frequent instruments</h3>
      {_list_pills(summary['inst_counts']) or "<span class='muted'>—</span>"}

      <h3>Phases (by frame count)</h3>
      {_list_pills(summary['phase_counts']) or "<span class='muted'>—</span>"}

      <h3>Common targets</h3>
      {_list_pills(summary['targ_counts']) or "<span class='muted'>—</span>"}

      <h3>Common actions</h3>
      {_list_pills(summary['action_counts']) or "<span class='muted'>—</span>"}
    </div>
    """

def _render_table_html(rows: List[Dict[str, str]], insights: List[str]) -> str:
    head = """
    <div class="card">
      <h2>Per-frame Predictions & Insights</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Phase</th>
            <th>Instrument(s)</th>
            <th>Target(s)</th>
            <th>Action</th>
            <th>MAP (mmHg)</th>
            <th>SpO₂ (%)</th>
            <th>ETCO₂ (mmHg)</th>
            <th>Insight</th>
          </tr>
        </thead>
        <tbody>
    """
    body = []
    for i, r in enumerate(rows):
        body.append(f"""
          <tr>
            <td class="mono">{i}</td>
            <td>{_escape(r.get('Phase',''))}</td>
            <td>{_escape(r.get('Instrument',''))}</td>
            <td>{_escape(r.get('Target',''))}</td>
            <td>{_escape(r.get('Surgical_Action',''))}</td>
            <td class="mono">{_escape(r.get('MAP_mmHg',''))}</td>
            <td class="mono">{_escape(r.get('SpO2_pct',''))}</td>
            <td class="mono">{_escape(r.get('ETCO2_mmHg',''))}</td>
            <td>{_escape(insights[i] if i < len(insights) else '')}</td>
          </tr>
        """)
    tail = """
        </tbody>
      </table>
      <div class="foot">Note: Insights are guidance-only. Not for diagnosis or treatment decisions.</div>
    </div>
    """
    return head + "\n".join(body) + tail

def _html_shell(inner: str, title: str) -> str:
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{_escape(title)}</title>
    {_inline_css()}
  </head>
  <body>
    <div class="container">
      <h1 class="title">{_escape(title)}</h1>
      <div class="subtle">Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
      {inner}
    </div>
  </body>
</html>"""

def save_report(rows: List[Dict[str, str]],
                insights: List[str],
                output_dir: str,
                video_name: Optional[str] = None) -> Dict[str, str]:
    """
    Creates output_dir and writes:
      - report.html
      - preds.csv
      - summary.json
      - (optional) report.pdf if WeasyPrint is available
    Returns dict of file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1) CSV
    csv_path = str(Path(output_dir) / "preds.csv")
    cols = ["Frame","Phase","Instrument","Target","Surgical_Action","MAP_mmHg","SpO2_pct","ETCO2_mmHg","Insight"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for i, r in enumerate(rows):
            line = [
                str(i),
                (r.get("Phase","").replace(",",";")),
                (r.get("Instrument","").replace(",",";")),
                (r.get("Target","").replace(",",";")),
                (r.get("Surgical_Action","").replace(",",";")),
                r.get("MAP_mmHg",""),
                r.get("SpO2_pct",""),
                r.get("ETCO2_mmHg",""),
                (insights[i] if i < len(insights) else "").replace("\n"," ").replace(",",";"),
            ]
            f.write(",".join(line) + "\n")

    # 2) Summary JSON
    summary = summarize_rows(rows)
    meta = {"video_name": video_name or "—"}
    summary_path = str(Path(output_dir) / "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "meta": meta}, f, indent=2)

    # 3) HTML
    html_inner = _render_summary_html(summary, meta) + _render_table_html(rows, insights)
    html = _html_shell(html_inner, "Cholecystectomy Post-operative Report")
    html_path = str(Path(output_dir) / "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 4) Optional PDF (if weasyprint exists & system libs available)
    pdf_path = ""
    try: # import here so import-time never crashes
        pdf_path = str(Path(output_dir) / "report.pdf")
        HTML(string=html).write_pdf(pdf_path)
    except Exception:
        pdf_path = ""  # silently skip if not installed / missing dependencies

    out = {
        "csv": csv_path,
        "summary": summary_path,
        "html": html_path,
    }
    if pdf_path:
        out["pdf"] = pdf_path
    return out