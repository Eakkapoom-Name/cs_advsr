
import os
import re
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import faiss  # type: ignore

from config import EMBED_MODEL

# Local embedding model (fast & free). Change to LiteLLM embeddings if desired.
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from rapidfuzz import fuzz
    _HAS_RF = True
except Exception:
    _HAS_RF = False


@dataclass
class IndexBundle:
    index: faiss.IndexFlatIP
    id2row: Dict[int, int]   # vector id -> df row index
    embeddings_dim: int
    model_name: str
    fingerprint: str


def file_sha1(path: str) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def build_text_chunk(row: pd.Series) -> str:
    # Use ONLY columns that exist in cs_coursedata.xlsx
    parts = []
    for col in ["รหัสวิชา", "ชื่อวิชา", "หน่วยกิต", "ประเภทวิชา", "แผนการศึกษา",
                "ชั้นปีที่เรียน", "เทอมที่เปิดสอน", "เกี่ยวกับวิชา", "หมายเหตุ"]:
        if col in row and pd.notna(row[col]):
            parts.append(f"{col}: {row[col]}")
    return "\n".join(parts)


def _load_local_model():
    if not _HAS_ST:
        raise RuntimeError(
            "sentence-transformers is not installed. Please install it or switch to LiteLLM embeddings."
        )
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts: List[str], st_model=None) -> np.ndarray:
    if st_model is None:
        st_model = _load_local_model()
    embs = st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embs, dtype="float32")


def save_index(bundle: IndexBundle, index_path: str, meta_path: str):
    faiss.write_index(bundle.index, index_path)
    meta = {
        "id2row": bundle.id2row,
        "embeddings_dim": bundle.embeddings_dim,
        "model_name": bundle.model_name,
        "fingerprint": bundle.fingerprint,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)


def load_index(index_path: str, meta_path: str) -> Optional[IndexBundle]:
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return IndexBundle(
        index=index,
        id2row=meta["id2row"],
        embeddings_dim=meta["embeddings_dim"],
        model_name=meta["model_name"],
        fingerprint=meta.get("fingerprint", ""),
    )


def ensure_index(df: pd.DataFrame, excel_path: str, cache_dir: str) -> IndexBundle:
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, "cs_index.faiss")
    meta_path = os.path.join(cache_dir, "cs_index_meta.pkl")
    fingerprint = file_sha1(excel_path)

    bundle = load_index(index_path, meta_path)
    st_model = None
    if bundle and bundle.fingerprint == fingerprint:
        # happy path
        return bundle

    # (re)build
    st_model = _load_local_model()
    texts = [build_text_chunk(row) for _, row in df.iterrows()]
    embs = embed_texts(texts, st_model)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    id2row = {i: int(idx) for i, idx in enumerate(df.index)}
    bundle = IndexBundle(index=index, id2row=id2row, embeddings_dim=dim,
                         model_name=EMBED_MODEL, fingerprint=fingerprint)
    save_index(bundle, index_path, meta_path)
    return bundle


def _keyword_score(q: str, row: pd.Series) -> float:
    """Lightweight keyword/fuzzy score in [0,1]."""
    q = (q or "").strip()
    if not q:
        return 0.0
    text_fields = []
    for col in ["ชื่อวิชา", "เกี่ยวกับวิชา", "หมายเหตุ", "รหัสวิชา"]:
        if col in row and pd.notna(row[col]):
            text_fields.append(str(row[col]))
    hay = " ".join(text_fields)

    # Exact/substring boosts
    score = 0.0
    if q in hay:
        score += 0.6

    # Token overlap
    tokens = [t for t in re.split(r"\s+|[,/()\-:]", q) if t]
    matches = sum(1 for t in tokens if t and t in hay)
    score += min(0.3, 0.1 * matches)

    # Fuzzy match (Thai-friendly approximate)
    if _HAS_RF:
        try:
            score += 0.01 * max(
                fuzz.partial_ratio(q, hay),
                fuzz.token_set_ratio(q, hay)
            )
        except Exception:
            pass
    return min(1.0, score)


def _faiss_candidates(bundle: IndexBundle, query: str, st_model=None, topn: int = 50) -> List[int]:
    if st_model is None:
        st_model = _load_local_model()
    q_emb = embed_texts([query], st_model)
    D, I = bundle.index.search(q_emb, topn)  # cosine-sim due to normalized embeddings
    return [int(i) for i in I[0] if i != -1]


def hybrid_search(query: str, df: pd.DataFrame, bundle: IndexBundle, st_model=None, k: int = 5, alpha: float = 0.6) -> List[Tuple[int, float]]:
    """
    Returns list of (row_index, score). alpha weights semantic vs keyword.
    1) FAISS to get topN candidates
    2) Re-rank with: score = alpha*semantic + (1-alpha)*keyword
    """
    if not query.strip():
        # Return top courses arbitrarily (e.g., required ones) if no query.
        base = df.index.tolist()
        return [(int(idx), 0.0) for idx in base[:k]]

    # FAISS candidates -> map to df row indexes
    cand_vec_ids = _faiss_candidates(bundle, query, st_model=st_model, topn=max(50, k*10))
    cand_rows = [bundle.id2row.get(cid, None) for cid in cand_vec_ids]
    cand_rows = [r for r in cand_rows if r is not None and r in df.index]

    # Compute scores
    # Semantic score ~= FAISS distances in first search result (rescale to [0,1])
    # For simplicity, assign a decaying semantic score by rank if distances not at hand.
    sem_scores = {r: 1.0 - (i / max(1, len(cand_rows))) for i, r in enumerate(cand_rows)}
    scores = []
    for r in cand_rows:
        kscore = _keyword_score(query, df.loc[r])
        final = alpha * sem_scores.get(r, 0.0) + (1 - alpha) * kscore
        scores.append((r, float(final)))

    # In case nothing from FAISS passed, fallback to keyword-only over all
    if not scores:
        all_scores = []
        for r in df.index.tolist():
            all_scores.append((r, _keyword_score(query, df.loc[r])))
        scores = sorted(all_scores, key=lambda x: x[1], reverse=True)[:k]

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    return scores


def parse_year_term_plan(text: str) -> Dict[str, Optional[str]]:
    # Extract "ปี X", "เทอม Y", and plan names (ปกติ/สหกิจ/ก้าวหน้า)
    info = {"year": None, "term": None, "plan": None}
    m_year = re.search(r"ปี\s*([1-4])", text)
    if m_year:
        info["year"] = m_year.group(1)
    m_term = re.search(r"เทอม\s*([1-3])", text)
    if m_term:
        info["term"] = m_term.group(1)
    # Plan keywords
    if "ก้าวหน้า" in text:
        info["plan"] = "ก้าวหน้า"
    elif "สหกิจ" in text:
        info["plan"] = "สหกิจ"
    else:
        info["plan"] = "ปกติ"
    return info


def filter_df(df: pd.DataFrame, year: Optional[str]=None, term: Optional[str]=None, plan: Optional[str]=None) -> pd.DataFrame:
    out = df.copy()
    if year and "ชั้นปีที่เรียน" in out.columns:
        out = out[out["ชั้นปีที่เรียน"].astype(str).str.contains(str(year), na=False)]
    if term and "เทอมที่เปิดสอน" in out.columns:
        out = out[out["เทอมที่เปิดสอน"].astype(str).str.contains(str(term), na=False)]
    if plan and "แผนการศึกษา" in out.columns:
        out = out[out["แผนการศึกษา"].astype(str).str.contains(plan, na=False)]
    return out


def extract_interests(text: str) -> List[str]:
    # very light heuristic: keep Thai words >= 3 chars, drop common filter words
    stop = set(["แนะนำ","ช่วย","จัด","เทอม","ปี","วิชา","ลง","ควร","อะไร","ที่","สำหรับ","และ","ให้","แบบ","แผน","ปกติ","สหกิจ","ก้าวหน้า"])
    tokens = re.split(r"[^0-9A-Za-zก-๙]+", text)
    cands = [t for t in tokens if len(t) >= 3 and t not in stop]
    return cands[:5]


def recommend_courses(df: pd.DataFrame, interests: List[str], limit: int = 8) -> pd.DataFrame:
    # Rank: required first ("บังคับ" in ประเภทวิชา), then interest match in เกี่ยวกับวิชา/ชื่อวิชา
    def interest_score(row):
        hay = " ".join([str(row.get("เกี่ยวกับวิชา","")), str(row.get("ชื่อวิชา",""))])
        s = 0
        for kw in interests:
            if kw and kw in hay:
                s += 1
        return s

    df = df.copy()
    df["__required"] = df["ประเภทวิชา"].astype(str).str.contains("บังคับ|แกน", na=False)
    df["__is_match"] = df.apply(interest_score, axis=1)
    df = df.sort_values(by=["__required","__is_match"], ascending=[False, False])
    return df.drop(columns=["__required","__is_match"]).head(limit)


def format_course_row(row: pd.Series) -> str:
    fields = []
    for col in ["รหัสวิชา","ชื่อวิชา","หน่วยกิต","ประเภทวิชา","แผนการศึกษา","ชั้นปีที่เรียน","เทอมที่เปิดสอน"]:
        if col in row and pd.notna(row[col]):
            fields.append(f"{col}: {row[col]}")
    if "เกี่ยวกับวิชา" in row and pd.notna(row["เกี่ยวกับวิชา"]):
        fields.append(f"เกี่ยวกับวิชา: {row['เกี่ยวกับวิชา']}")
    if "หมายเหตุ" in row and pd.notna(row["หมายเหตุ"]):
        fields.append(f"หมายเหตุ: {row['หมายเหตุ']}")
    return "\n".join(fields)


def build_context(df: pd.DataFrame, hits: List[Tuple[int,float]]) -> str:
    ctx = []
    for idx, score in hits:
        row = df.loc[idx]
        ctx.append(format_course_row(row))
    return "\n\n---\n\n".join(ctx)


def search_courses_by_topic(df: pd.DataFrame, topic: str, bundle: IndexBundle, limit: int = 10) -> pd.DataFrame:
    """
    Search courses by topic/keyword using hybrid search.
    Returns DataFrame of matching courses.
    """
    st_model = _load_local_model()
    hits = hybrid_search(topic, df, bundle, st_model=st_model, k=limit, alpha=0.7)
    
    if not hits:
        return pd.DataFrame()
    
    result_indices = [idx for idx, _ in hits]
    return df.loc[result_indices].copy()


def find_prerequisite_courses(df: pd.DataFrame, course_code: str) -> pd.DataFrame:
    """
    Find courses that require the given course_code as prerequisite.
    Checks 'หมายเหตุ' column for courses that contain this course code.
    """
    if 'หมายเหตุ' not in df.columns:
        return pd.DataFrame()
    
    # Clean course code (remove spaces, convert to string)
    course_code = str(course_code).strip()
    
    # Find courses where หมายเหตุ contains this course code
    mask = df['หมายเหตุ'].astype(str).str.contains(course_code, na=False, regex=False)
    result = df[mask].copy()
    
    return result