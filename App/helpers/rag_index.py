# App/helpers/rag_index.py  (TF-IDF, no Torch/FAISS; macOS-safe)
from __future__ import annotations
import os, glob, json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PyPDF2 import PdfReader

# ---- RAG chunking ----
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
MAX_CHUNKS = 1200  # hard cap to keep memory bounded

@dataclass
class RAGIndex:
    base: str                 # cache base WITHOUT extension (e.g., ".../Preprocessed_Files/llm_rag_index")
    texts: List[str]
    meta: List[Tuple[str, int]]

# --------- tiny TF-IDF (no external deps) ----------
def _tokenize(s: str) -> List[str]:
    out, buf = [], []
    for ch in s.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out

def _build_tfidf(texts: List[str], base: str):
    """
    Build a sparse TF-IDF CSR from texts.
    Persist arrays next to `base`:
      base.data.npy, base.indices.npy, base.indptr.npy, base.vocab.json, base.idf.npy
    """
    # vocab
    vocab = {}
    docs_tokens = []
    for t in texts:
        tok = _tokenize(t)
        docs_tokens.append(tok)
        for w in tok:
            if w not in vocab:
                vocab[w] = len(vocab)

    V = len(vocab)
    N = len(texts)
    if V == 0 or N == 0:
        raise ValueError("Empty corpus after tokenization.")

    # term frequencies per doc (CSR)
    data, indices, indptr = [], [], [0]
    df = np.zeros(V, dtype=np.int32)

    for tok in docs_tokens:
        counts = {}
        for w in tok:
            j = vocab[w]
            counts[j] = counts.get(j, 0) + 1
        for j in counts.keys():
            df[j] += 1
        for j, c in sorted(counts.items()):
            data.append(float(c))
            indices.append(int(j))
        indptr.append(len(data))

    data = np.array(data, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    indptr = np.array(indptr, dtype=np.int32)

    # smoothed IDF
    df = df.astype(np.float32)
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0

    # TF*IDF then L2-normalize rows
    data *= idf[indices]
    norms = np.zeros(N, dtype=np.float32)
    for i in range(N):
        s, e = indptr[i], indptr[i+1]
        norms[i] = np.sqrt(np.sum(data[s:e]**2)) + 1e-8
        data[s:e] /= norms[i]

    # save
    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.save(base + ".data.npy",   data)
    np.save(base + ".indices.npy",indices)
    np.save(base + ".indptr.npy", indptr)
    with open(base + ".vocab.json", "w") as f:
        json.dump(vocab, f)
    np.save(base + ".idf.npy",    idf.astype(np.float32))

def _load_tfidf(base: str):
    data    = np.load(base + ".data.npy")
    indices = np.load(base + ".indices.npy")
    indptr  = np.load(base + ".indptr.npy")
    with open(base + ".vocab.json", "r") as f:
        vocab = json.load(f)
    idf     = np.load(base + ".idf.npy")
    return (data, indices, indptr, vocab, idf)

def _tfidf_query_vector(q: str, vocab: dict, idf: np.ndarray):
    tok = _tokenize(q)
    if not tok:
        return {}, 1.0
    tf = {}
    for w in tok:
        j = vocab.get(w, -1)
        if j >= 0:
            tf[j] = tf.get(j, 0.0) + 1.0
    if not tf:
        return {}, 1.0
    for j in list(tf.keys()):
        tf[j] *= float(idf[j])
    norm = np.sqrt(np.sum([v*v for v in tf.values()])) + 1e-8
    for j in list(tf.keys()):
        tf[j] /= norm
    return tf, norm

def _cosine_topk_tfidf(data, indices, indptr, qvec: dict, k: int, N: int):
    # naive sparse dot: iterate rows; fine for <= few thousand chunks
    scores = np.zeros(N, dtype=np.float32)
    # pre-extract q columns/values for speed
    q_cols, q_vals = zip(*qvec.items()) if qvec else ([], [])
    for i in range(N):
        s, e = indptr[i], indptr[i+1]
        cols = indices[s:e]
        vals = data[s:e]
        if cols.size == 0:
            continue
        # intersect with q columns
        # small segments: linear is OK
        sc = 0.0
        for qc, qv in zip(q_cols, q_vals):
            # find qc in cols
            hit = np.where(cols == qc)[0]
            if hit.size:
                sc += float(vals[hit[0]]) * float(qv)
        scores[i] = sc

    if k >= N:
        top = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k)[:k]
        top = idx[np.argsort(-scores[idx])]
    return top, scores[top]
# -------------------------------------------------

def _read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def _chunk(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    i, L = 0, len(text)
    while i < L and len(chunks) < MAX_CHUNKS:
        j = min(i + size, L)
        ch = text[i:j].strip()
        if ch:
            chunks.append(ch)
        i = j - overlap
        if i <= 0:
            i = j
    return chunks

def build_index(pdf_dir: str, cache_base: str) -> RAGIndex:
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found at: {pdf_dir}")

    texts, meta = [], []
    for p in pdf_paths:
        if len(texts) >= MAX_CHUNKS:
            break
        t = _read_pdf_text(p)
        for k, ch in enumerate(_chunk(t, CHUNK_SIZE, CHUNK_OVERLAP)):
            texts.append(ch)
            meta.append((os.path.basename(p), k))
            if len(texts) >= MAX_CHUNKS:
                break

    # build & persist TF-IDF at cache_base.*
    _build_tfidf(texts, cache_base)
    # also persist texts/meta next to same base
    np.save(cache_base + ".texts.npy", np.array(texts, dtype=object))
    np.save(cache_base + ".meta.npy",  np.array(meta,  dtype=object))
    return RAGIndex(base=cache_base, texts=texts, meta=meta)

def load_or_build_index(pdf_dir: str, cache_path: str) -> RAGIndex:
    """
    cache_path is an *extensionless* base (e.g., ".../Preprocessed_Files/llm_rag_index").
    """
    # strip any accidental extensions
    base = cache_path
    while True:
        b2, ext = os.path.splitext(base)
        if not ext:
            break
        base = b2

    # ensure parent exists
    os.makedirs(os.path.dirname(base), exist_ok=True)

    needed = [
        base + ".data.npy",
        base + ".indices.npy",
        base + ".indptr.npy",
        base + ".vocab.json",
        base + ".idf.npy",
        base + ".texts.npy",
        base + ".meta.npy",
    ]
    if all(os.path.exists(p) for p in needed):
        texts = list(np.load(base + ".texts.npy", allow_pickle=True))
        meta  = list(map(tuple, np.load(base + ".meta.npy", allow_pickle=True)))
        return RAGIndex(base=base, texts=texts, meta=meta)

    # (re)build
    return build_index(pdf_dir, base)

def retrieve(rag: RAGIndex, query: str, k: int = 5) -> List[Tuple[str, Tuple[str,int], float]]:
    data, indices, indptr, vocab, idf = _load_tfidf(rag.base)
    N = len(rag.texts)
    if N == 0:
        return []
    qvec, _ = _tfidf_query_vector(query, vocab, idf)
    if not qvec:
        return []

    top_idx, top_scores = _cosine_topk_tfidf(data, indices, indptr, qvec, k, N)
    out = []
    for idx, score in zip(top_idx, top_scores):
        out.append((rag.texts[int(idx)], rag.meta[int(idx)], float(score)))
    return out