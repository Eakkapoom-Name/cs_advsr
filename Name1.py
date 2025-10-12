import os
import streamlit as st
import pandas as pd
import base64
import json
import time
from pathlib import Path

from utils.llm_client import LLMClient, get_available_models

from rag_core import (
    ensure_index, hybrid_search, filter_df, parse_year_term_plan,
    extract_interests, recommend_courses, build_context, format_course_row
)

from config import MODEL
from litellm import completion

EXCEL_PATH = "/mnt/c/Users/petez/OneDrive/Documents/dawg_sheet/cs_coursedata.xlsx"
st.set_page_config(page_title="Study Helper (RAG Chat)",page_icon="📚",layout="wide")

#image
def get_base64_image(image_file):
    if not os.path.exists(image_file):
        return ""
        
    try:
        with open(image_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""
    
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    if "is_model_ready" not in st.session_state:
        st.session_state.is_model_ready = False
    if "is_rag_ready" not in st.session_state:
        st.session_state.is_rag_ready = False
# Main
def main():
    init_session_state()
    
    #background
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, "png", "background1.png")
    base64_encoded = get_base64_image(image_path)
    if base64_encoded:
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_encoded}");
            background-position: center; 
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover; 
            background-color: #000000;
            background-image: none !important;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    header_col1, header_col2, header_col3 = st.columns([1, 10, 1])
    
    #title center
    title_color = "#000000"
    
    with header_col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <h1 style="color: {title_color};">📚 Study Helper</h1>
            <p>This is a RAG-powered Study Helper for CS CMU projects.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # ✅ ใช้ path ที่อิงจากไฟล์ app3.py
            current_dir = os.path.dirname(__file__)
            logo_path = os.path.join(current_dir, "png", "cslogo2.png")

            if os.path.exists(logo_path):
                st.image(logo_path, width=100)
            else:
                st.warning(f"⚠️ ไม่พบโลโก้: {logo_path}")

if __name__ == "__main__":
    main()


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
    bundle = ensure_index(df, excel_path, cache_dir=cache_dir)
    # we don't cache the st_model to keep memory light; ensure_index returns cached on disk
    return bundle

def call_llm(system_prompt: str, user_prompt: str, model_name: str, max_tokens: int = 600):
    resp = completion(
        model=model_name,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return str(resp)

def system_prompt_th():
    return (
        "คุณเป็นที่ปรึกษาวิชาเรียนของภาควิชาวิทยาการคอมพิวเตอร์ ช่วยตอบคำถามโดยอ้างอิงเฉพาะข้อมูลในบริบท (context) ที่ให้ไว้เท่านั้น "
        "ตอบเป็นภาษาไทย กระชับ ชัดเจน และครบถ้วน ถ้าพบข้อมูลหลายวิชาควรสรุปเป็นรายการหัวข้อและข้อมูลไม่ขาดตกโดยไม่สร้างข้อมูลที่ไม่มีในบริบท"
    )

def build_user_prompt(question: str, context: str):
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

def system_prompt_recommend_th():
    return (
        "คุณเป็นที่ปรึกษาวิชาเรียนของภาควิชาวิทยาการคอมพิวเตอร์ "
        "หน้าที่ของคุณคือเลือกและเรียบเรียงคำแนะนำรายวิชาอย่างเป็นธรรมชาติ "
        "โดยอ้างอิงได้เฉพาะวิชาที่อยู่ในบริบท (Context) เท่านั้น ห้ามสร้างรายวิชาใหม่ "
        "ให้รักษาลำดับตามที่บริบทให้มา (วิชาบังคับมาก่อน) "
        "อธิบายเหตุผลสั้น ๆ ว่าทำไมจึงเหมาะกับผู้ใช้ และสรุปให้กระชับแต่ข้อมูลครบถ้วน"
    )

def apply_filters(df, year, term, plan, course_type):
    year_v = None if year == "ทั้งหมด" else year
    term_v = None if term == "ทั้งหมด" else term
    plan_v = None if plan == "ทั้งหมด" else plan
    tmp = filter_df(df, year_v, term_v, plan_v)
    if course_type != "ทั้งหมด" and "ประเภทวิชา" in tmp.columns:
        if course_type == "วิชาบังคับ":
            tmp = tmp[tmp["ประเภทวิชา"].astype(str).str.contains("บังคับ|แกน", na=False)]
        elif course_type == "วิชาเลือก":
            tmp = tmp[tmp["ประเภทวิชา"].astype(str).str.contains("เลือก", na=False)]

    return tmp

def answer_general(question: str):
    # Hybrid search on filtered df
    hits = hybrid_search(question, df_filtered, bundle, k=topk, alpha=alpha)
    if not hits:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในข้อมูลที่กรองไว้"

    context = build_context(df_filtered, hits)
    sys = system_prompt_th()
    user = build_user_prompt(question, context)
    return call_llm(sys, user, model_name=selected_model, max_tokens=max_tokens)

def answer_recommend(question: str, num_of_course: int = 5):
    info = parse_year_term_plan(question)
    tmp = df
    # Enforce filter as spec: filter BEFORE RAG/recommend
    tmp = apply_filters(tmp,
                        info["year"] if info["year"] else year,
                        info["term"] if info["term"] else term,
                        info["plan"] if info["plan"] else plan,
                        course_type)
    if tmp.empty:
        return "ไม่มีวิชาที่ตรงตามเงื่อนไขที่กรอง"

    interests = extract_interests(question)
    recs = recommend_courses(tmp, interests, limit=8)

    lines = []
    if interests:
        lines.append(f"**ความสนใจที่จับได้:** {', '.join(interests)}")
    lines.append("**รายการแนะนำ:**")
    course_counter = 0
    for _, row in recs.iterrows():
        if course_counter < num_of_course:
            bullet = f"- {row['รหัสวิชา']} {row['ชื่อวิชา']}\n• {row['หน่วยกิต']} หน่วยกิต | {row['ประเภทวิชา']} | เทอมที่เปิดสอน : เทอม {row['เทอมที่เปิดสอน']} | แผน: {row['แผนการศึกษา']}"
            if str(row.get("เกี่ยวกับวิชา","")).strip():
                bullet += f"\n  • เกี่ยวกับวิชา: {row['เกี่ยวกับวิชา']}"
            if str(row.get("หมายเหตุ","")).strip():
                bullet += f"\n  • หมายเหตุ: {row['หมายเหตุ']}"
            lines.append(bullet)
            course_counter += 1
        else:
            break
    lines.append("\n*หากต้องการจัดลำดับใหม่หรือระบุจำนวนที่ต้องการ ให้พิมพ์เช่น 'เอา 5 วิชา' หรือเพิ่มคีย์เวิร์ดความสนใจ*")
    return "\n".join(lines)

def answer_recommend_llm(question: str):
    # 1) parse ปี/เทอม/แผนจากคำถาม
    info = parse_year_term_plan(question)

    # 2) กรองก่อน (รวมค่าจาก UI ถ้ายังไม่ได้ระบุในคำถาม)
    tmp = apply_filters(
        df,
        info["year"] if info["year"] else year,
        info["term"] if info["term"] else term,
        info["plan"] if info["plan"] else plan,
        course_type
    )
    if tmp.empty:
        return "ไม่มีวิชาที่ตรงตามเงื่อนไขกรอง"

    # 3) ดึงคีย์เวิร์ดความสนใจแล้วจัดอันดับ
    interests = extract_interests(question)
    # เลือกมากหน่อยเพื่อให้ LLM มีบริบทเพียงพอ แต่ไม่ยาวเกินไป
    recs = recommend_courses(tmp, interests, limit=12)

    # (ทางเลือก) ผสมผล hybrid_search เพิ่มอีกเล็กน้อย
    # hits = hybrid_search(" ".join(interests) or question, tmp, bundle, k=5, alpha=alpha)
    # ctx_hits = build_context(tmp, hits)

    # 4) สร้าง Context จากรายการ recs ที่จัดแล้ว (บังคับมาก่อน)
    lines = []
    for _, row in recs.iterrows():
        lines.append(format_course_row(row))
    context = "\n\n---\n\n".join(lines)
    # context = context + ("\n\n" + ctx_hits if ctx_hits else "")  # ถ้าต้องการผสม

    # 5) Prompt เข้า LLM (กำกับให้ใช้เฉพาะวิชาใน Context)
    sys = system_prompt_recommend_th()
    guide = (
        "จงแนะนำรายวิชาที่เหมาะสม โดยรักษาลำดับตามที่ปรากฏใน Context "
        "(วิชาบังคับมาก่อน จากนั้นวิชาที่เกี่ยวข้องกับความสนใจถ้ามี) "
        "ห้ามกล่าวถึงวิชาที่ไม่มีใน Context "
        "หาก Context มีข้อมูล 'เกี่ยวกับวิชา' หรือ 'หมายเหตุ' ให้ใช้เพื่ออธิบายเหตุผลแบบสั้นๆ.\n"
        f"ความสนใจที่จับได้: {', '.join(interests) if interests else '—'}"
    )
    user = (
        f"{guide}\n\n### Context\n{context}\n\n"
        f"### คำถามผู้ใช้\n{question}\n\n"
        "รูปแบบคำตอบ: เล่าเป็นภาษาธรรมชาติ 1-2 ย่อหน้า แล้วตามด้วย bullet สรุป (1) บังคับ (2) เกี่ยวข้อง"
    )

    return call_llm(sys, user, model_name=selected_model, max_tokens=max_tokens)


def is_recommend_intent(text: str):
    keys = ["จัด","ตอน","วิชา","ลงวิชา","เรียนวิชา","วิชาอะไร","เรียนอะไร","เกี่ยวกับอะไร","คืออะไร","ช่วย","ควรลง","วางแผน","แผนการเรียน","ต้องผ่าน","ทั้งหมด","บังคับ","เลือก","แผนปกติ","แผนสหกิจ","แผนก้าวหน้า","ปกติ","สหกิจ","ก้าวหน้า"]
    return any(k in text for k in keys)

def toggle_mandatory():
    st.session_state.show_mandatory = not st.session_state.show_mandatory


# Sidebar
with st.sidebar:
    st.header("⚙️ การตั้งค่า")

    # Filter UI And Clear Chat
    with st.expander("ตัวกรองข้อมูล", expanded=False):
        year = st.selectbox("ชั้นปีที่เรียน", ("ทั้งหมด", "1", "2", "3", "4"))
        term = st.selectbox("เทอมที่เปิดสอน", ("ทั้งหมด", "1", "2"))
        plan = st.selectbox("แผนการศึกษา", ("ทั้งหมด", "แผนปกติ", "แผนสหกิจ", "แผนก้าวหน้า"))
        course_type = st.selectbox("ประเภทวิชา", ("ทั้งหมด", "วิชาบังคับ", "วิชาเลือก"))

    # ปรับRAG
    with st.expander("ปรับค่าค้นหา", expanded=False):
        temperature = st.slider("Temperature",min_value=0.1,max_value=1.0,value=0.4,step=0.1,help="ขอบเขตของการสุ่ม")
        max_tokens = st.slider("Max Tokens", min_value=500, max_value=2500, value=1500, step=100,help="จำนวน Token ที่ใช้ได้")
        alpha = st.slider("ค่า Alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.05,help="Alpha เท่ากับ 1 จะใช้ Vector อย่างเดียวแต่ถ้า เท่ากับ 0 จะใช้ Keyword")
        topk = st.slider("ค่า k", min_value=5, max_value=15, value=10, step=1,help="จำนวนเอกสารอ้างอิง")
    st.divider()
    available_models = get_available_models()
    selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=1,
            help="เลือกโมเดล"
        )

    with st.spinner("Initializing model..."):
            st.session_state.llm_client = LLMClient(
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens)
    # ตัวกรอง
    st.info(f"⚡Model {selected_model} initialized!")
    # st.markdown(selected_model)
    use_llm_rec = st.checkbox("ใช้ LLM เรียบเรียงคำแนะนำ (RAG)", value=True)
    
    st.divider()
    #Clear Chat
    if st.button("ล้างประวัติแชท"):
        st.session_state.messages = []
        st.rerun()
# Load data + index
df = load_data(EXCEL_PATH)
bundle = get_index_bundle(df, EXCEL_PATH)

df_filtered = apply_filters(df, year, term, plan, course_type)

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"สวัสดี! พิมพ์คำถามได้เลย เช่น \"ปี 2 เทอม 2 เรียนวิชาอะไรบ้าง\" หรือ \"แนะนำวิชาปี 3 เทอม 1 ที่เกี่ยวเกี่ยวกับ AI\""}
    ]

# Chat history UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
q = st.chat_input("พิมพ์คำถามที่นี่...")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            if is_recommend_intent(q):
                ans = answer_recommend_llm(q) if use_llm_rec else answer_recommend(q)
            else:
                ans = answer_general(q) 
            st.markdown(ans)
            st.session_state.messages.append({"role":"assistant","content":ans})

if "show_mandatory" not in st.session_state:
    st.session_state.show_mandatory = False


# ตารางกรอง
st.markdown(f"**จำนวนวิชาหลังกรอง:** {len(df_filtered)} รายการ")
with st.expander("ดูตารางหลังกรอง"):
    showed_cols = ["รหัสวิชา", "ชื่อวิชา", "หน่วยกิต", "เทอมที่เปิดสอน"]
    showed_cols = [c for c in showed_cols if c in df_filtered.columns]
    st.dataframe(df_filtered[showed_cols].reset_index(drop=True))


# Example
cA, cB, cC = st.columns(3)
with cA:
    button_A = st.button("ปี1 เรียนอะไรบ้าง")

with cB:
    button_B =  st.button("วิชา 204271 ต้องผ่านตัวอะไรบ้าง")

with cC:
    st.download_button(
        "ดาวน์โหลดผลกรองเป็น CSV",
        data=df_filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_courses.csv",
        mime="text/csv"
        )
if button_A:
    q = "ปี1 เรียนอะไรบ้าง"
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if is_recommend_intent(q):
            ans = answer_recommend_llm(q) if use_llm_rec else answer_recommend(q)
        else:
            ans = answer_general(q)
        st.markdown(ans)
        st.session_state.messages.append({"role":"assistant","content":ans})

if button_B:
    q = "วิชา 204271 ต้องผ่านตัวอะไรบ้าง"
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if is_recommend_intent(q):
            ans = answer_recommend_llm(q) if use_llm_rec else answer_recommend(q)
        else:
            ans = answer_general(q)
        st.markdown(ans)
        st.session_state.messages.append({"role":"assistant","content":ans})