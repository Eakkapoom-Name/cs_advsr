
# 🎓 LLM Course Advisor (Thai) — RAG + FAISS + LiteLLM + Streamlit

แอปตัวอย่างที่:
- อ่านข้อมูลรายวิชาจาก `cs_coursedata.xlsx` ด้วย **pandas**
- สร้าง **RAG** ด้วย FAISS + Embedding (Sentence-Transformers) + Hybrid Search (semantic + keyword/fuzzy)
- ใช้ **LiteLLM** เป็นตัวเรียก LLM
- ทำ UI แบบ Chatbot ด้วย **Streamlit** พร้อมเก็บ chat history
- **Filter ก่อน** ค่อยส่งเข้าค้นหา/แนะนำ ตามสเปกของคุณ

## โครงสร้างไฟล์

```
course_advisor_app/
├── app.py
├── rag_core.py
├── requirements.txt
└── (คุณต้องมี) /mnt/data/cs_coursedata.xlsx
```

> โค้ดนี้อ้างอิงเฉพาะคอลัมน์ที่มีในไฟล์จริง: `รหัสวิชา, ชื่อวิชา, หน่วยกิต, หมายเหตุ, เกี่ยวกับวิชา, เทอมที่เปิดสอน, ชั้นปีที่เรียน, ประเภทวิชา, แผนการศึกษา`

## วิธีใช้งาน

1) ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

2) ตั้งค่า API Key สำหรับ LiteLLM (เช่นใช้ OpenAI)
```bash
export OPENAI_API_KEY=...         # หรือ LITELLM_API_KEY=...
```

3) รันแอป
```bash
streamlit run app.py
```

4) ใน Sidebar ตั้ง path ไฟล์ Excel (ค่าเริ่มต้นคือ `/mnt/data/cs_coursedata.xlsx`) และตั้งชื่อโมเดล เช่น `gpt-4o-mini`

## การทำงานหลัก

- **Save/Load FAISS Index:** แอปจะสร้าง index ไว้ในโฟลเดอร์ `.faiss_cache` ข้างๆไฟล์ Excel และตรวจ fingerprint ของไฟล์ หากไฟล์ Excel เปลี่ยนจะ re-build อัตโนมัติ
- **Hybrid Search:** ใช้ FAISS (semantic) + ค่าคะแนน keyword/fuzzy (rapidfuzz) ผสมด้วย `alpha`
- **Filter ก่อน:** มีตัวกรอง ปี/เทอม/แผน/ประเภทวิชา เพื่อตัดให้เหลือเฉพาะวิชาที่รับได้ ก่อนจะส่งเข้าค้นหา/ตอบ/แนะนำ
- **แนะนำวิชา:** ถ้าพิมพ์ว่า *แนะนำ/จัด/ควรลง/วางแผน...* ระบบจะจับ intent และจัดลิสต์ **วิชาบังคับก่อน** แล้วค่อยตามด้วยวิชาที่ตรงความสนใจ
- **Q&A:** คำถามทั่วไปจะดึงบริบท top-k จากตารางที่กรองไว้ แล้วใช้ LLM (ผ่าน LiteLLM) ตอบ

## ปรับแต่ง

- เปลี่ยน embedding เป็นของ LiteLLM ได้ โดยแก้ใน `rag_core.py` ฟังก์ชัน `embed_texts` ให้เรียก `litellm.embedding()` และตั้งชื่อโมเดล embedding
- หากต้องการ BM25 จริงจัง ให้ติดตั้ง `rank-bm25` และคำนวณคะแนนแทนฟังก์ชัน `_keyword_score`

## หมายเหตุ

- โค้ดพยายามไม่ใช้อะไรนอกเหนือจากคอลัมน์ที่ไฟล์มีจริง
- ภาษาเริ่มต้นเป็นภาษาไทย
