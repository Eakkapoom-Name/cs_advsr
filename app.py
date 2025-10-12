
import os
import json
import time
import streamlit as st
import pandas as pd

from typing import Any

from rag_core import (
    ensure_index, hybrid_search, filter_df, parse_year_term_plan,
    extract_interests, recommend_courses, build_context, format_course_row, find_prerequisite_courses, find_prerequisite_direct
)
from config import MODEL

# LiteLLM
from litellm import completion

EXCEL_PATH = "cs_coursedata.xlsx"

st.set_page_config(page_title="AI Course Advisor", page_icon="🎓", layout="wide")

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

def call_llm(system_prompt: str, user_prompt: str, model_name: str, max_tokens: int = 600) -> str:
    resp = completion(
        model=model_name,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.4,
        max_tokens=max_tokens
    )
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return str(resp)

def system_prompt_th():
    return (
        "คุณเป็นที่ปรึกษาวิชาเรียนของภาควิชาวิทยาการคอมพิวเตอร์ ช่วยตอบคำถามโดยอ้างอิงเฉพาะข้อมูลในบริบท (context) ที่ให้ไว้เท่านั้น "
        "ตอบเป็นภาษาไทย กระชับ ชัดเจน และถ้าพบข้อมูลหลายวิชาควรสรุปเป็นรายการหัวข้อ โดยไม่สร้างข้อมูลที่ไม่มีในบริบท "
        "คุณมี tools ที่สามารถใช้ได้:\n"
        "1. search_courses_by_topic - ค้นหาวิชาตามหัวข้อ/เนื้อหา\n"
        "2. find_prerequisite_courses - หาวิชาที่ต้องใช้วิชาที่ระบุเป็นตัวต่อ\n"
        "ใช้ tools เหล่านี้เพื่อตอบคำถามอย่างถูกต้อง แล้วอธิบายคำตอบเป็นภาษาไทยที่เข้าใจง่าย "
        "ห้ามสร้างข้อมูลที่ไม่มีในผลลัพธ์จาก tools"
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
        "หากคำถามถามถึงตัวต่อให้ดึงจากคอลัมน์ 'หมายเหตุ' ที่รหัสวิชาตรงกับวิชาที่ถาม \n"
    )
    return guide

def system_prompt_recommend_th():
    return (
        "คุณเป็นที่ปรึกษาวิชาเรียนของภาควิชาวิทยาการคอมพิวเตอร์ "
        "หน้าที่ของคุณคือเลือกและเรียบเรียงคำแนะนำรายวิชาอย่างเป็นธรรมชาติ "
        "โดยอ้างอิงได้เฉพาะวิชาที่อยู่ในบริบท (Context) เท่านั้น ห้ามสร้างรายวิชาใหม่ "
        "ให้รักษาลำดับตามที่บริบทให้มา (วิชาบังคับมาก่อน) "
        "อธิบายเหตุผลสั้น ๆ ว่าทำไมจึงเหมาะกับผู้ใช้ และสรุปให้กระชับ"
    )

# Sidebar
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    # excel_path = st.sidebar.text_input("ที่อยู่ไฟล์ Excel", value="cs_coursedata.xlsx")
    # model_name = st.sidebar.text_input("LiteLLM Model", value="gpt-4o-mini")
    llm_token = st.slider("จำนวน Token ที่ใข้", min_value=800, max_value=2000, value=1500, step=50)
    alpha = st.slider("น้ำหนักการค้นหาเชิงความหมาย (alpha)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    topk = st.slider("จำนวนเอกสารอ้างอิง (k)", min_value=5, max_value=15, value=10, step=1)
    use_llm_rec = st.checkbox("ใช้ LLM เรียบเรียงคำแนะนำ (RAG)", value=True)

    st.caption("ตั้งค่า ENV เช่น OPENAI_API_KEY หรือ LITELLM_API_KEY ก่อนรันแอป")

# Main
st.title("🎓 LLM Course Advisor 🎓")
st.markdown(
    "ใช้ข้อมูลจากไฟล์ Excel และค้นหาแบบ **ไฮบริด** (semantic + keyword) พร้อม **กรองตามปี/เทอม/แผน** ก่อนจะส่งต่อให้ LLM"
)

# Load data + index
df = load_data(EXCEL_PATH)
bundle = get_index_bundle(df, EXCEL_PATH)

# Filter UI
with st.expander("ตัวกรองข้อมูล (ใช้ก่อน RAG)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    year = c1.selectbox("ชั้นปีที่เรียน", ("ทั้งหมด", "1", "2", "3", "4"))
    term = c2.selectbox("เทอมที่เปิดสอน", ("ทั้งหมด", "1", "2"))
    plan = c3.selectbox("แผนการศึกษา", ("ทั้งหมด", "แผนปกติ", "แผนสหกิจ", "แผนก้าวหน้า"))
    course_type = c4.selectbox("ประเภทวิชา", ("ทั้งหมด", "วิชาบังคับ", "วิชาเลือก"))

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

df_filtered = apply_filters(df, year, term, plan, course_type)

st.markdown(f"**จำนวนวิชาหลังกรอง:** {len(df_filtered)} รายการ")
with st.expander("ดูตารางหลังกรอง"):
    showed_cols = ["รหัสวิชา", "ชื่อวิชา", "หน่วยกิต", "เทอมที่เปิดสอน"]
    showed_cols = [c for c in showed_cols if c in df_filtered.columns]
    st.dataframe(df_filtered[showed_cols].reset_index(drop=True))

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"สวัสดี! พิมพ์คำถามได้เลย เช่น 'วิชา 204xxx มีกี่หน่วยกิต' หรือ 'แนะนำวิชาปี 3 เทอม 1 เกี่ยวกับ AI'"}
    ]

# Chat history UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# เพิ่ม tool definition หลังจาก load_data
def get_tools_definition():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_courses_by_topic",
                "description": "ค้นหาวิชาที่มีเนื้อหาเกี่ยวข้องกับหัวข้อหรือคีย์เวิร์ดที่กำหนด เช่น AI, hardware, database, web development",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "หัวข้อหรือคีย์เวิร์ดที่ต้องการค้นหา เช่น 'AI', 'machine learning', 'hardware', 'network'"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "จำนวนวิชาที่ต้องการแสดงผล",
                            "default": 5
                        }
                    },
                    "required": ["topic"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_prerequisite_courses",
                "description": "หาวิชาที่ต้องเรียนวิชาที่ระบุเป็นตัวต่อ (prerequisite) โดยเช็คในคอลัมน์หมายเหตุว่ามีรหัสวิชาใดบ้างที่ต้องเรียนวิชานี้ก่อน",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_code": {
                            "type": "string",
                            "description": "รหัสวิชาที่ต้องการเช็คว่าเป็นตัวต่อของวิชาอะไรบ้าง เช่น '204111', '204101'"
                        }
                    },
                    "required": ["course_code"]
                }
            }
        }
    ]

# เพิ่ม function สำหรับเรียก LLM พร้อม tools
def call_llm_with_tools(system_prompt: str, user_prompt: str, model_name: str, tools: list, max_tokens: int = 800):
    """Call LLM with tool support"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    resp = completion(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.4,
        max_tokens=max_tokens
    )
    return resp

# เพิ่ม function สำหรับจัดการ tool calls
def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute the requested tool and return results"""
    
    if tool_name == "search_courses_by_topic":
        topic = arguments.get("topic", "")
        limit = arguments.get("limit", 5)
        
        from rag_core import search_courses_by_topic
        results = search_courses_by_topic(df_filtered, topic, bundle, limit=limit)
        
        if results.empty:
            return f"ไม่พบวิชาที่เกี่ยวข้องกับ '{topic}'"
        
        lines = [f"พบ {len(results)} วิชาที่เกี่ยวข้องกับ '{topic}':\n"]
        for idx, row in results.iterrows():
            line = f"- {row['รหัสวิชา']} {row['ชื่อวิชา']} ({row['หน่วยกิต']} หน่วยกิต)"
            if pd.notna(row.get('เกี่ยวกับวิชา')):
                line += f"\n  • {row['เกี่ยวกับวิชา'][:150]}..."
            lines.append(line)
        
        return "\n".join(lines)
    
    elif tool_name == "find_prerequisite_courses":
        course_code = arguments.get("course_code", "")
        
        from rag_core import find_prerequisite_courses
        results = find_prerequisite_courses(df_filtered, course_code)
        
        if results.empty:
            return f"ไม่พบวิชาที่ต้องใช้ {course_code} เป็นวิชาตัวต่อ (prerequisite)"
        
        lines = [f"วิชาที่ต้องเรียน {course_code} เป็นตัวต่อมี {len(results)} วิชา:\n"]
        for idx, row in results.iterrows():
            line = f"- {row['รหัสวิชา']} {row['ชื่อวิชา']}"
            if pd.notna(row.get('หมายเหตุ')):
                line += f"\n  • หมายเหตุ: {row['หมายเหตุ']}"
            if pd.notna(row.get('เกี่ยวกับวิชา')):
                line += f"\n  • เกี่ยวกับ: {row['เกี่ยวกับวิชา'][:100]}..."
            lines.append(line)
        
        return "\n".join(lines)
    
    return f"ไม่รู้จัก tool: {tool_name}"


def answer_general_with_tools(question: str) -> str:
    """Answer with tool calling support"""
    tools = get_tools_definition()
    sys = system_prompt_th()
    
    # First call to LLM
    resp = call_llm_with_tools(sys, question, MODEL, tools, max_tokens=llm_token)
    
    # Check if LLM wants to use tools
    message = resp["choices"][0]["message"]
    
    if message.get("tool_calls"):
        # Execute tool calls
        tool_results = []
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            result = execute_tool_call(func_name, arguments)
            tool_results.append(result)
        
        # Second call with tool results
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": question},
            message,
            {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], 
             "content": "\n\n".join(tool_results)}
        ]
        
        final_resp = completion(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=llm_token
        )
        return final_resp["choices"][0]["message"]["content"]
    
    # No tool calls, return direct response
    return answer_general(question)

def answer_general(question: str) -> str:
    # Hybrid search on filtered df
    hits = hybrid_search(question, df_filtered, bundle, k=topk, alpha=alpha)
    if not hits:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในบริบทที่กรองไว้"

    context = build_context(df_filtered, hits)
    sys = system_prompt_th()
    user = build_user_prompt(question, context)
    return call_llm(sys, user, model_name=MODEL, max_tokens=llm_token)

def answer_recommend(question: str, num_of_course:int = 5) -> str:
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
    return "\n".join(lines)

def answer_recommend_llm(question: str) -> str:
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

    # 3) ดึงคีย์เวิร์ดความสนใจ แล้วจัดอันดับผู้ท้าชิง
    interests = extract_interests(question)
    # เลือกมากหน่อยเพื่อให้ LLMมีบริบทเพียงพอ แต่ไม่ยาวเกินไป
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
        "หาก Context มีข้อมูล 'เกี่ยวกับวิชา' หรือ 'หมายเหตุ' ให้ใช้เพื่ออธิบายเหตุผลแบบสั้น ๆ.\n"
        f"ความสนใจที่จับได้: {', '.join(interests) if interests else '—'}"
    )
    user = (
        f"{guide}\n\n### Context\n{context}\n\n"
        f"### คำถามผู้ใช้\n{question}\n\n"
        "รูปแบบคำตอบ: เล่าเป็นภาษาธรรมชาติ 1-2 ย่อหน้า แล้วตามด้วย bullet สรุป (1) บังคับ (2) เกี่ยวข้อง"
    )

    return call_llm(sys, user, model_name=MODEL, max_tokens=llm_token)

def is_recommend_intent(text: str) -> bool:
    keys = ["จัด","ลงวิชา","ควรลง","วางแผน","แผนการเรียน", "แนะนำ"]
    return any(k in text for k in keys)


# Chat input
q = st.chat_input("พิมพ์คำถามที่นี่...")
if q:
    st.session_state.messages.append({"role":"user","content":q})

    with st.chat_message("assistant"):
        if is_recommend_intent(q):
            ans = answer_recommend_llm(q) if use_llm_rec else answer_recommend(q)
        else:
            ans = answer_general_with_tools(q)

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
