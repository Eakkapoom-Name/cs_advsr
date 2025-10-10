
import os
import json
import time
import streamlit as st
import pandas as pd

from rag_core import (
    ensure_index, hybrid_search, filter_df, parse_year_term_plan,
    extract_interests, recommend_courses, build_context, format_course_row
)
from config import MODEL

# LiteLLM
from litellm import completion

EXCEL_PATH = "cs_coursedata.xlsx"

st.set_page_config(page_title="ที่ปรึกษาวิชาเรียน (RAG+FAISS)", page_icon="🎓", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(excel_path: str):
    df = pd.read_excel(excel_path)
    # Keep only the known columns (as in the file)
    cols = ['รหัสวิชา','ชื่อวิชา','หน่วยกิต','หมายเหตุ','เกี่ยวกับวิชา','เทอมที่เปิดสอน','ชั้นปีที่เรียน','ประเภทวิชา','แผนการศึกษา']
    df = df[[c for c in cols if c in df.columns]].copy()
    # Normalize types for robust filtering
    for c in ["รหัสวิชา","ชื่อวิชา","หมายเหตุ","เกี่ยวกับวิชา","ประเภทวิชา","แผนการศึกษา"]:
        if c in df.columns: df[c] = df[c].astype(str)
    for c in ["เทอมที่เปิดสอน","ชั้นปีที่เรียน","หน่วยกิต"]:
        if c in df.columns: df[c] = df[c].astype(str)
    return df

@st.cache_resource(show_spinner=True)
def get_index_bundle(df, excel_path: str):
    # cache_dir in the app folder
    cache_dir = os.path.join(os.path.dirname(excel_path), ".faiss_cache")
    bundle, st_model = ensure_index(df, excel_path, cache_dir=cache_dir)
    # we don't cache the st_model to keep memory light; ensure_index returns cached on disk
    return bundle

def call_llm(system_prompt: str, user_prompt: str, model_name: str, max_tokens: int = 600) -> str:
    resp = completion(
        model=model_name,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.2,
        max_tokens=max_tokens
    )
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return str(resp)

def system_prompt_th():
    return (
        "คุณเป็นที่ปรึกษาวิชาเรียนของภาควิชาวิทยาการคอมพิวเตอร์ ช่วยตอบคำถามโดยอ้างอิงเฉพาะข้อมูลในบริบท (context) ที่ให้ไว้เท่านั้น "
        "ตอบเป็นภาษาไทย กระชับ ชัดเจน และถ้าพบข้อมูลหลายวิชาควรสรุปเป็นรายการหัวข้อ โดยไม่สร้างข้อมูลที่ไม่มีในบริบท"
    )

def build_user_prompt(question: str, context: str) -> str:
    guide = (
        "คำถามของผู้ใช้อยู่ด้านล่าง ให้ตอบโดยอิงจากบริบทเท่านั้น:\n\n"
        "### บริบท\n"
        f"{context}\n\n"
        "### คำถาม\n"
        f"{question}\n\n"
        "หากคำถามถามถึงจำนวนหน่วยกิต/วิชาบังคับ/วิชาที่เกี่ยวข้อง ให้ดึงจากคอลัมน์ที่ตรงในบริบท:\n"
        "- หน่วยกิต: คอลัมน์ 'หน่วยกิต'\n"
        "- ประเภทวิชา (บังคับ/เลือก): คอลัมน์ 'ประเภทวิชา'\n"
        "- เกี่ยวกับเนื้อหา: คอลัมน์ 'เกี่ยวกับวิชา'\n"
        "- เงื่อนไข/ข้อควรรู้: 'หมายเหตุ'\n"
    )
    return guide

# Sidebar
st.sidebar.header("⚙️ การตั้งค่า")
# excel_path = st.sidebar.text_input("ที่อยู่ไฟล์ Excel", value="cs_coursedata.xlsx")
# model_name = st.sidebar.text_input("LiteLLM Model", value="gpt-4o-mini")
alpha = st.sidebar.slider("น้ำหนักการค้นหาเชิงความหมาย (alpha)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
topk = st.sidebar.slider("จำนวนเอกสารอ้างอิง (k)", min_value=3, max_value=15, value=5, step=1)

st.sidebar.caption("ตั้งค่า ENV เช่น OPENAI_API_KEY หรือ LITELLM_API_KEY ก่อนรันแอป")

# Main
st.title("🎓 LLM Course Advisor — RAG + FAISS + LiteLLM")
st.markdown(
    "ใช้ข้อมูลจากไฟล์ Excel และค้นหาแบบ **ไฮบริด** (semantic + keyword) พร้อม **กรองตามปี/เทอม/แผน** ก่อนจะส่งต่อให้ LLM"
)

# Load data + index
df = load_data(EXCEL_PATH)
bundle = get_index_bundle(df, EXCEL_PATH)

# Filter UI
with st.expander("ตัวกรองข้อมูล (ใช้ก่อน RAG)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    year = c1.selectbox("ชั้นปีที่เรียน", ["ทั้งหมด"] + sorted({str(x) for x in df["ชั้นปีที่เรียน"].dropna().unique()}))
    term = c2.selectbox("เทอมที่เปิดสอน", ["ทั้งหมด"] + sorted({str(x) for x in df["เทอมที่เปิดสอน"].dropna().unique()}))
    plan = c3.selectbox("แผนการศึกษา", ["ทั้งหมด"] + sorted({str(x) for x in df["แผนการศึกษา"].dropna().unique()}))
    course_type = c4.selectbox("ประเภทวิชา", ["ทั้งหมด"] + sorted({str(x) for x in df["ประเภทวิชา"].dropna().unique()}))

def apply_filters(df, year, term, plan, course_type):
    year_v = None if year == "ทั้งหมด" else year
    term_v = None if term == "ทั้งหมด" else term
    plan_v = None if plan == "ทั้งหมด" else plan
    tmp = filter_df(df, year_v, term_v, plan_v)
    if course_type != "ทั้งหมด" and "ประเภทวิชา" in tmp.columns:
        tmp = tmp[tmp["ประเภทวิชา"].astype(str).str.contains(course_type, na=False)]
    return tmp

df_filtered = apply_filters(df, year, term, plan, course_type)

st.markdown(f"**จำนวนวิชาหลังกรอง:** {len(df_filtered)} รายการ")
with st.expander("ดูตารางหลังกรอง"):
    st.dataframe(df_filtered.reset_index(drop=True))

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"สวัสดี! พิมพ์คำถามได้เลย เช่น 'วิชา 204xxx มีกี่หน่วยกิต' หรือ 'แนะนำวิชาปี 3 เทอม 1 เกี่ยวกับ AI'"}
    ]

# Chat history UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def answer_general(question: str) -> str:
    # Hybrid search on filtered df
    hits = hybrid_search(question, df_filtered, bundle, k=topk, alpha=alpha)
    if not hits:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในบริบทที่กรองไว้"

    context = build_context(df_filtered, hits)
    sys = system_prompt_th()
    user = build_user_prompt(question, context)
    return call_llm(sys, user, model_name=MODEL)

def answer_recommend(question: str) -> str:
    info = parse_year_term_plan(question)
    tmp = df
    # Enforce filter as spec: filter BEFORE RAG/recommend
    tmp = apply_filters(tmp,
                        info["year"] if info["year"] else year,
                        info["term"] if info["term"] else term,
                        info["plan"] if info["plan"] else plan,
                        course_type)
    if tmp.empty:
        return "ไม่มีวิชาที่ตรงตามเงื่อนไขกรอง"

    interests = extract_interests(question)
    recs = recommend_courses(tmp, interests, limit=8)

    lines = []
    if interests:
        lines.append(f"**ความสนใจที่จับได้:** {', '.join(interests)}")
    lines.append("**รายการแนะนำ (เรียงวิชาบังคับก่อน):**")
    for _, row in recs.iterrows():
        bullet = f"- {row['รหัสวิชา']} {row['ชื่อวิชา']} — {row['หน่วยกิต']} หน่วยกิต | {row['ประเภทวิชา']} | ปี {row['ชั้นปีที่เรียน']} เทอม {row['เทอมที่เปิดสอน']} | แผน: {row['แผนการศึกษา']}"
        if str(row.get("เกี่ยวกับวิชา","")).strip():
            bullet += f"\n  • เกี่ยวกับวิชา: {row['เกี่ยวกับวิชา']}"
        if str(row.get("หมายเหตุ","")).strip():
            bullet += f"\n  • หมายเหตุ: {row['หมายเหตุ']}"
        lines.append(bullet)
    lines.append("\n*หากต้องการจัดลำดับใหม่หรือระบุจำนวนที่ต้องการ ให้พิมพ์เช่น 'เอา 5 วิชา' หรือเพิ่มคีย์เวิร์ดความสนใจ*")
    return "\n".join(lines)

def is_recommend_intent(text: str) -> bool:
    keys = ["แนะนำ","จัด","ลงวิชา","ควรลง","วางแผน","แผนการเรียน"]
    return any(k in text for k in keys)

# Chat input
q = st.chat_input("พิมพ์คำถามที่นี่...")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("assistant"):
        if is_recommend_intent(q):
            ans = answer_recommend(q)
        else:
            ans = answer_general(q)
        st.markdown(ans)
        st.session_state.messages.append({"role":"assistant","content":ans})

# Tools
cA, cB, cC = st.columns([1,1,1])
with cA:
    if st.button("ล้างประวัติแชท"):
        st.session_state.messages = []
        st.rerun()
with cB:
    if st.button("ดูวิชา 'บังคับ' เท่านั้น (หลังกรอง)"):
        if "ประเภทวิชา" in df_filtered.columns:
            sub = df_filtered[df_filtered["ประเภทวิชา"].astype(str).str.contains("บังคับ", na=False)]
            st.dataframe(sub.reset_index(drop=True))
        else:
            st.info("ไม่มีคอลัมน์ 'ประเภทวิชา'")
with cC:
    st.download_button(
        "ดาวน์โหลดผลกรองเป็น CSV",
        data=df_filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_courses.csv",
        mime="text/csv"
    )
