import os
import json
import streamlit as st
import pandas as pd
import base64

from pathlib import Path
from utils.llm_client import LLMClient, get_available_models
from typing import Any

from rag_core import (
    ensure_index, hybrid_search, filter_df, parse_year_term_plan,
    extract_interests, recommend_courses, build_context, format_course_row,
    search_courses_by_topic, find_prerequisite_courses
)

from config import MODEL
from litellm import completion

EXCEL_PATH = "cs_coursedata.xlsx"


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


def load_css(styles: str):
    with open(styles, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(excel_path: str):
    df = pd.read_excel(excel_path)
    cols = ['รหัสวิชา', 'ชื่อวิชา', 'หน่วยกิต', 'หมายเหตุ', 'เกี่ยวกับวิชา', 'เทอมที่เปิดสอน', 'ชั้นปีที่เรียน',
            'ประเภทวิชา', 'แผนการศึกษา']
    df = df[[c for c in cols if c in df.columns]].copy()
    for c in ["รหัสวิชา", "ชื่อวิชา", "หมายเหตุ", "เกี่ยวกับวิชา", "ประเภทวิชา", "แผนการศึกษา"]:
        if c in df.columns: df[c] = df[c].astype(str)
    for c in ["เทอมที่เปิดสอน", "ชั้นปีที่เรียน", "หน่วยกิต"]:
        if c in df.columns: df[c] = df[c].astype(str)
    return df


@st.cache_resource(show_spinner=True)
def get_index_bundle(df, excel_path: str):
    # cache_dir in the app folder
    cache_dir = os.path.join(os.path.dirname(excel_path), ".faiss_cache")
    bundle = ensure_index(df, excel_path, cache_dir=cache_dir)
    return bundle


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


def call_llm(system_prompt: str, user_prompt: str, model_name: str, max_tokens: int = 600):
    resp = completion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
        "คุณมี tools ที่สามารถใช้ได้:\n"
        "1. search_courses_by_topic - ค้นหาวิชาตามหัวข้อ/เนื้อหา\n"
        "2. find_prerequisite_courses - หาวิชาที่ต้องใช้วิชาที่ระบุเป็นตัวต่อ\n"
        "ใช้ tools เหล่านี้เพื่อตอบคำถามอย่างถูกต้อง แล้วอธิบายคำตอบเป็นภาษาไทยที่เข้าใจง่าย "
        "ห้ามสร้างข้อมูลที่ไม่มีในผลลัพธ์จาก tools"
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
        "หากคำถามถามถึงตัวต่อให้ดึงจากคอลัมน์ 'หมายเหตุ' ที่รหัสวิชาตรงกับวิชาที่ถาม \n"
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
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp


def execute_tool_call(tool_name: str, arguments: dict):
    """Execute the requested tool and return results"""

    if tool_name == "search_courses_by_topic":
        topic = arguments.get("topic", "")
        limit = arguments.get("limit", 5)

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


def answer_general_with_tools(question: str, tool_name: str = "") -> str:
    """Answer with tool calling support"""
    tools = get_tools_definition()
    sys = system_prompt_th()

    resp = call_llm_with_tools(sys, question, MODEL, tools, max_tokens=max_tokens)

    message = resp["choices"][0]["message"]

    if message.get("tool_calls"):
        # Execute tool calls
        # print("tool called")
        tool_results = []
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            result = execute_tool_call(func_name, arguments)
            tool_results.append(result)

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
            temperature=temperature,
            max_tokens=max_tokens
        )
        return final_resp["choices"][0]["message"]["content"]

    return answer_general(question)


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
    # Enforce filter as spec filter BEFORE RAG recommend
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
            if str(row.get("เกี่ยวกับวิชา", "")).strip():
                bullet += f"\n  • เกี่ยวกับวิชา: {row['เกี่ยวกับวิชา']}"
            if str(row.get("หมายเหตุ", "")).strip():
                bullet += f"\n  • หมายเหตุ: {row['หมายเหตุ']}"
            lines.append(bullet)
            course_counter += 1
        else:
            break
    return "\n".join(lines)


def answer_recommend_llm(question: str):
    info = parse_year_term_plan(question)

    tmp = apply_filters(
        df,
        info["year"] if info["year"] else year,
        info["term"] if info["term"] else term,
        info["plan"] if info["plan"] else plan,
        course_type
    )
    if tmp.empty:
        return "ไม่มีวิชาที่ตรงตามเงื่อนไขกรอง"
    interests = extract_interests(question)

    recs = recommend_courses(tmp, interests, limit=12)

    lines = []
    for _, row in recs.iterrows():
        lines.append(format_course_row(row))
    context = "\n\n---\n\n".join(lines)

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
    keys = ["จัด", "ตอน", "วิชา", "ลงวิชา", "เรียนวิชา", "วิชาอะไร", "เรียนอะไร", "เกี่ยวกับอะไร", "คืออะไร", "ช่วย",
            "ควรลง", "วางแผน", "แผนการเรียน", "ต้องผ่าน", "ทั้งหมด", "บังคับ", "เลือก", "แผนปกติ", "แผนสหกิจ",
            "แผนก้าวหน้า", "ปกติ", "สหกิจ", "ก้าวหน้า"]
    return any(k in text for k in keys)


def is_topic_tool_intent(text: str):
    keys = ["หัวข้อ", "เกี่ยว", "เกี่ยวกับ", "คล้าย", "เรื่อง", "เกี่ยวข้อง"]
    t = text.lower()
    return any(k in t for k in keys)


def is_prereq_tool_intent(text: str):
    keys = ["ตัวต่อ", "ต่อจาก", "ต่อ", "ต้องใช้", "ตัวต่อของวิชา", "ใช้"]
    t = text.lower()
    return any(k in t for k in keys)


def main():
    global max_tokens, alpha, topk, use_llm_rec, selected_model, temperature
    global df, bundle, df_filtered, year, term, plan, course_type

    # title
    st.set_page_config(
        page_title="Courses Advisor",
        page_icon="📚",
        layout="wide"
    )

    init_session_state()
    load_css("styles.css")

    header_col1, header_col2, header_col3 = st.columns([1, 10, 1])

    with header_col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 class="gradient-title" style="font-size: 3.5em; line-height: 0.5; font-weight: 700;">
                Courses Advisor
            </h1>
            <p class="subtitle" style="font-size: 1.5em; font-weight: 600;">
                This is a RAG-powered Courses Adviser for CS CMU projects.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            current_dir = os.path.dirname(__file__)
            logo_path = os.path.join(current_dir, "png", "cslogowhite.png")

            if os.path.exists(logo_path):
                st.image(logo_path, width=100)
            else:
                st.warning(f"⚠️ ไม่พบโลโก้: {logo_path}")
    # Sidebar
    with st.sidebar:
        st.header("⚙️ การตั้งค่า")

        # Filter UI And Clear Chat
        with st.expander("ตัวกรองข้อมูล", expanded=False):
            year = st.selectbox("ชั้นปีที่เรียน", ("ทั้งหมด", "1", "2", "3", "4"))
            term = st.selectbox("เทอมที่เปิดสอน", ("ทั้งหมด", "1", "2"))
            plan = st.selectbox("แผนการศึกษา", ("ทั้งหมด", "แผนปกติ", "แผนสหกิจ", "แผนก้าวหน้า"))
            course_type = st.selectbox("ประเภทวิชา", ("ทั้งหมด", "วิชาบังคับ", "วิชาเลือก"))

        # ปรับ RAG
        with st.expander("ค่าการค้นหา", expanded=False):
            temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.4, step=0.1,
                                    help="ขอบเขตของการสุ่ม")
            max_tokens = st.slider("Max Tokens", min_value=500, max_value=2500, value=1500, step=100,
                                   help="จำนวน Token ที่ใช้")
            alpha = st.slider("ค่า Alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                              help="Alpha เท่ากับ 1 จะใช้ Vector อย่างเดียวแต่ถ้า เท่ากับ 0 จะใช้ Keyword")
            topk = st.slider("ค่า k", min_value=5, max_value=15, value=10, step=1, help="จำนวนเอกสารอ้างอิง")

        st.divider()
        available_models = get_available_models()
        selected_model = st.selectbox(
            "เลือกโมเดล",
            available_models,
            index=1,
            help="เลือกโมเดลที่ต้องการใช้"
        )

        with st.spinner("Initializing model..."):
            st.session_state.llm_client = LLMClient(
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens)

        # ตัวกรอง
        st.info(f"⚡ กำลังใช้โมเดล {selected_model}")
        # st.markdown(selected_model)
        use_llm_rec = st.checkbox("ใช้ LLM เรียบเรียงคำแนะนำ (RAG)", value=True)

        st.divider()

        # Clear Chat
        if st.button("🗑️ ล้างประวัติการแชท"):
            st.session_state.messages = []
            st.rerun()

    # Load data + index
    df = load_data(EXCEL_PATH)
    bundle = get_index_bundle(df, EXCEL_PATH)

    df_filtered = apply_filters(df, year, term, plan, course_type)

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "สวัสดี! พิมพ์คำถามได้เลย เช่น \"ปี 2 เทอม 2 เรียนวิชาอะไรบ้าง\" หรือ \"แนะนำวิชาปี 3 เทอม 1 ที่เกี่ยวเกี่ยวกับ AI\""}
        ]

    # Chat history UI
    for msg in st.session_state.messages:
        avatar = msg.get("avatar", "png/avatar_bot.png")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    q = st.chat_input("พิมพ์คำถามที่นี่...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q, "avatar": "png/avatar_user.png"})
        with st.chat_message("user", avatar="png/avatar_user.png"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="png/avatar_bot.png"):
            with st.spinner("Processing..."):
                if is_topic_tool_intent(q):
                    ans = answer_general_with_tools(q, "search_courses_by_topic")

                elif is_prereq_tool_intent(q):
                    ans = answer_general_with_tools(q, "find_prerequisite_courses")

                elif is_recommend_intent(q):
                    ans = answer_recommend_llm(q) if use_llm_rec else answer_recommend(q)

                else:
                    ans = answer_general_with_tools(q)
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans, "avatar": "png/avatar_bot.png"})

    # ตารางกรอง
    st.markdown(f"**จำนวนวิชาหลังกรอง:** {len(df_filtered)} รายการ")
    with st.expander("ดูตารางหลังกรอง"):
        showed_cols = ["รหัสวิชา", "ชื่อวิชา", "หน่วยกิต", "เทอมที่เปิดสอน"]
        showed_cols = [c for c in showed_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[showed_cols].reset_index(drop=True))

    # Example
    cA, cB, cC = st.columns(3)
    with cA:
        button_A = st.button("วิชาที่เกี่ยวกับ AI")

    with cB:
        button_B = st.button("วิชาใดบ้างที่เป็นตัวต่อของ 204252")

    with cC:
        st.download_button(
            "ดาวน์โหลดผลกรองเป็น CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_courses.csv",
            mime="text/csv"
        )
    if button_A:
        q = "วิชาที่เกี่ยวกับ AI"
        st.session_state.messages.append({"role": "user", "content": q, "avatar": "png/avatar_user.png"})
        with st.chat_message("user", avatar="png/avatar_user.png"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="png/avatar_bot.png"):
            with st.spinner("Processing..."):
                if is_topic_tool_intent(q):
                    ans = answer_general_with_tools(q, "search_courses_by_topic")

                elif is_prereq_tool_intent(q):
                    ans = answer_general_with_tools(q, "find_prerequisite_courses")

                elif is_recommend_intent(q):
                    ans = answer_recommend_llm(q) if use_llm_rec else answer_recommend(q)

                else:
                    ans = answer_general_with_tools(q)
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans, "avatar": "png/avatar_bot.png"})

    if button_B:
        q = "วิชาใดบ้างที่เป็นตัวต่อของ 204252"
        st.session_state.messages.append({"role": "user", "content": q, "avatar": "png/avatar_user.png"})
        with st.chat_message("user", avatar="png/avatar_user.png"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="png/avatar_bot.png"):
            with st.spinner("Processing..."):
                if is_topic_tool_intent(q):
                    ans = answer_general_with_tools(q, "search_courses_by_topic")

                elif is_prereq_tool_intent(q):
                    ans = answer_general_with_tools(q, "find_prerequisite_courses")

                elif is_recommend_intent(q):
                    ans = answer_recommend_llm(q) if use_llm_rec else answer_recommend(q)

                else:
                    ans = answer_general_with_tools(q)
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans, "avatar": "png/avatar_bot.png"})


if __name__ == "__main__":
    main()
