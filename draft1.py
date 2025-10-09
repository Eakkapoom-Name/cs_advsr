import pandas as pd
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
import litellm
import json
import re
from config import MODEL

class RAGSystem:
    """
    ระบบ RAG สำหรับข้อมูลรายวิชา
    ใช้ FAISS สำหรับการค้นหาแบบ Vector และ LiteLLM สำหรับการสร้างคำตอบ
    """
    
    def __init__(self):
        """
        Initialize RAG System
        
        Args:
            model_name: ชื่อโมเดล LLM ที่จะใช้ผ่าน LiteLLM
        """
        # 1. กำหนดโมเดลสำหรับแปลงข้อความเป็น embeddings
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # 2. กำหนด LLM model
        self.llm_model = MODEL
        
        # 3. ตัวแปรเก็บข้อมูล
        self.df = None
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        อ่านไฟล์ Excel และทำความสะอาดข้อมูล
        
        Args:
            file_path: path ของไฟล์ Excel
            
        Returns:
            DataFrame ที่ผ่านการทำความสะอาด
        """
        print("📖 กำลังอ่านไฟล์ Excel...")
        
        # อ่านไฟล์ Excel
        df = pd.read_excel(file_path)
        
        # ทำความสะอาดชื่อคอลัมน์ (ลบ space และอักขระพิเศษ)
        df.columns = df.columns.str.strip()
        
        # แปลงค่า NaN เป็น empty string
        df = df.fillna('')
        
        # เก็บ DataFrame
        self.df = df
        
        print(f"✅ อ่านข้อมูลสำเร็จ: {len(df)} รายวิชา")
        print(f"📊 คอลัมน์ที่พบ: {df.columns.tolist()}")
        
        return df
    
    def prepare_documents(self) -> List[Dict]:
        """
        เตรียมข้อมูลเอกสารสำหรับการสร้าง embeddings
        แปลงแต่ละแถวของ DataFrame เป็นเอกสารที่มีข้อความและ metadata
        
        Returns:
            List ของ documents พร้อม metadata
        """
        print("\n📝 กำลังเตรียมเอกสาร...")
        
        documents = []
        
        for idx, row in self.df.iterrows():
            # สร้างข้อความจากข้อมูลที่สำคัญ
            # ปรับตามชื่อคอลัมน์จริงในไฟล์ของคุณ
            text_parts = []
            
            # รวมข้อมูลสำคัญเป็นข้อความ
            if 'รหัสวิชา' in row and row['รหัสวิชา']:
                text_parts.append(f"รหัสวิชา: {row['รหัสวิชา']}")
            
            if 'ชื่อวิชา' in row and row['ชื่อวิชา']:
                text_parts.append(f"ชื่อวิชา: {row['ชื่อวิชา']}")
            
            if 'หน่วยกิต' in row and row['หน่วยกิต']:
                text_parts.append(f"หน่วยกิต: {row['หน่วยกิต']}")
            
            if 'หมายเหตุ' in row and row['หมายเหตุ']:
                text_parts.append(f"หมายเหตุ: {row['หมายเหตุ']}")
            
            if 'เกี่ยวกับวิชา' in row and row['เกี่ยวกับวิชา']:
                text_parts.append(f"รายละเอียด: {row['เกี่ยวกับวิชา']}")

            if 'เทอมที่เปิดสอน' in row and row['เทอมที่เปิดสอน']:
                text_parts.append(f"เทอมที่เปิดสอน: {row['เทอมที่เปิดสอน']}")

            if 'ชั้นปีที่เรียน' in row and row['ชั้นปีที่เรียน']:
                text_parts.append(f"ชั้นปีที่เรียน: {row['ชั้นปีที่เรียน']}")

            if 'ประเภทวิชา' in row and row['ประเภทวิชา']:
                text_parts.append(f"ประเภทวิชา: {row['ประเภทวิชา']}")

            if 'แผนการศึกษา' in row and row['แผนการศึกษา']:
                text_parts.append(f"แผนการศึกษา: {row['แผนการศึกษา']}")

            # รวมข้อความทั้งหมด
            full_text = " ".join(text_parts)
            
            # สร้าง document dictionary
            doc = {
                'id': idx,
                'text': full_text,
                'metadata': row.to_dict()  # เก็บข้อมูลทั้งหมดใน metadata
            }
            
            documents.append(doc)
        
        self.documents = documents
        print(f"✅ เตรียมเอกสารสำเร็จ: {len(documents)} เอกสาร")
        
        return documents
    
    def create_embeddings(self) -> np.ndarray:
        """
        สร้าง embeddings จากเอกสารทั้งหมด
        
        Returns:
            numpy array ของ embeddings
        """
        print("\n🔄 กำลังสร้าง embeddings...")
        
        # ดึงเฉพาะข้อความจาก documents
        texts = [doc['text'] for doc in self.documents]
        
        # สร้าง embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # แปลงเป็น numpy array และ normalize
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.embeddings = embeddings
        print(f"✅ สร้าง embeddings สำเร็จ: shape {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self):
        """
        สร้าง FAISS index สำหรับการค้นหาแบบ vector similarity
        """
        print("\n🏗️ กำลังสร้าง FAISS index...")
        
        # ขนาดของ embedding dimension
        dimension = self.embeddings.shape[1]
        
        # สร้าง FAISS index แบบ Inner Product (หลัง normalize = cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        
        # เพิ่ม embeddings เข้า index
        self.index.add(self.embeddings)
        
        print(f"✅ สร้าง FAISS index สำเร็จ: {self.index.ntotal} vectors")
    
    def extract_year_term(self, query: str):
        """
        ตรวจจับปีการศึกษาและเทอมจากข้อความผู้ใช้
        เช่น 'ปี 3 เทอม 1', 'ปีสอง', 'เทอมสอง', ฯลฯ
        """
        year = None
        term = None
        plan = None

        # จับปี
        year_match = re.search(r'ปี\s*([1-4])', query)
        if year_match:
            year = int(year_match.group(1))

        # จับเทอม
        term_match = re.search(r'เทอม\s*([1-2])', query)
        if term_match:
            term = int(term_match.group(1))

        plan_match = re.search(r'(แผนปกติ|แผนก้าวหน้า|แผนสหกิจ)', query)
        if plan_match:
            plan = plan_match.group(1).strip()

        return year, term, plan

    def search(self, query: str, k: int = 5) -> List[Dict]:
        print(f"\n🔍 กำลังค้นหา: '{query}'")

        # ✅ ตรวจจับปี/เทอม
        year, term, plan = self.extract_year_term(query)

        filtered_docs = self.documents

        # ✅ ถ้ามีปีหรือเทอม ให้กรอง metadata ก่อน
        if any([year, term, plan]):
            def match_meta(doc):
                meta = doc['metadata']
                conds = []
                if year:
                    conds.append(str(year) in str(meta.get('ชั้นปีที่เรียน', '')).strip())
                if term:
                    conds.append(str(term) in str(meta.get('เทอมที่เปิดสอน', '')).strip())
                if plan:
                    conds.append(plan in str(meta.get('แผนการศึกษา', '')).strip())
                return all(conds) if conds else True
            filtered_docs = [d for d in self.documents if match_meta(d)]
            print(f"📚 กรองเหลือ {len(filtered_docs)} เอกสาร ตามปี/เทอม")

            # ถ้าไม่มีเอกสารตรงเงื่อนไข ให้ fallback กลับไปใช้ทั้งหมด
            if not filtered_docs:
                filtered_docs = self.documents

        # แปลง query เป็น embedding
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # สร้าง embeddings เฉพาะเอกสารที่เหลือ
        texts = [d['text'] for d in filtered_docs]
        doc_embeddings = self.encoder.encode(texts)
        doc_embeddings = np.array(doc_embeddings).astype('float32')
        faiss.normalize_L2(doc_embeddings)

        # สร้าง index ชั่วคราว
        temp_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        temp_index.add(doc_embeddings)

        # ค้นหา
        distances, indices = temp_index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                results.append({
                    'document': filtered_docs[idx],
                    'score': float(distances[0][i])
                })

        print(f"✅ พบเอกสารที่เกี่ยวข้อง {len(results)} รายการ")
        return results

    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        สร้างคำตอบโดยใช้ LLM กับบริบทที่ได้จากการค้นหา
        
        Args:
            query: คำถามจากผู้ใช้
            context_docs: เอกสารที่เกี่ยวข้องจากการค้นหา
            
        Returns:
            คำตอบที่สร้างโดย LLM
        """
        print("\n💭 กำลังสร้างคำตอบ...")
        
        # สร้างบริบทจากเอกสารที่ค้นหาได้
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            metadata = doc['document']['metadata']
            context_parts.append(f"เอกสารที่ {i}:")
            
            # แสดงข้อมูลสำคัญจาก metadata
            for key, value in metadata.items():
                if value and str(value).strip():  # ตรวจสอบว่ามีค่า
                    context_parts.append(f"- {key}: {value}")
            
            context_parts.append("")  # เพิ่มบรรทัดว่าง
        
        context = "\n".join(context_parts)
        
        # สร้าง prompt สำหรับ LLM
        prompt = f"""คุณเป็นผู้ช่วยที่มีความรู้เกี่ยวกับหลักสูตรและรายวิชา 
        โปรดตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น

        ข้อมูลรายวิชา:
        {context}

        คำถาม: {query}

        คำตอบ (ภาษาไทย):"""
        
        try:
            # เรียกใช้ LLM ผ่าน LiteLLM
            response = litellm.completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "คุณเป็นผู้ช่วยที่ให้ข้อมูลเกี่ยวกับรายวิชาอย่างถูกต้องและชัดเจน"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            print("✅ สร้างคำตอบสำเร็จ")
            return answer
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            return f"ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        ฟังก์ชันหลักสำหรับถาม-ตอบ
        
        Args:
            question: คำถามจากผู้ใช้
            k: จำนวนเอกสารที่จะใช้เป็นบริบท
            
        Returns:
            Dictionary ที่มีคำตอบและเอกสารอ้างอิง
        """
        # 1. ค้นหาเอกสารที่เกี่ยวข้อง
        search_results = self.search(question, k)
        
        # 2. สร้างคำตอบจากบริบท
        answer = self.generate_answer(question, search_results)
        
        # 3. จัดรูปแบบผลลัพธ์
        result = {
            'question': question,
            'answer': answer,
            'sources': search_results
        }
        
        return result
    
    def save_index(self, path: str):
        """บันทึก FAISS index"""
        faiss.write_index(self.index, path)
        print(f"💾 บันทึก index ที่: {path}")
    
    def load_index(self, path: str):
        """โหลด FAISS index"""
        self.index = faiss.read_index(path)
        print(f"📂 โหลด index จาก: {path}")


# ตัวอย่างการใช้งาน
def main():
    """
    ตัวอย่างการใช้งานระบบ RAG
    """
    # 1. สร้างระบบ RAG
    rag = RAGSystem()
    
    # 2. โหลดข้อมูลจาก Excel
    # แทนที่ path นี้ด้วย path จริงของไฟล์ Excel
    df = rag.load_excel_data("cs_coursedata.xlsx")
    
    # 3. เตรียมเอกสาร
    rag.prepare_documents()
    
    # 4. สร้าง embeddings
    rag.create_embeddings()
    
    # 5. สร้าง FAISS index
    rag.build_faiss_index()
    
    # 6. บันทึก index (optional)
    rag.save_index("course_index.faiss")
    
    # 7. ทดสอบระบบด้วยคำถาม
    questions = [
        "มีวิชาเกี่ยวกับ AI อะไรบ้าง?",
        "วิชา Database มีกี่หน่วยกิต?",
        "วิชาที่ต้องเรียนก่อนวิชา Machine Learning คืออะไร?",
        "ช่วยจัดวิชาที่ควรจะลงในถ้าศึกษาในปริญญาตรี ปี 3 เทอม 1"
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        result = rag.query(q, k=5)
        print(f"คำถาม: {result['question']}")
        print(f"คำตอบ: {result['answer']}")
        print(f"อ้างอิง: พบ {len(result['sources'])} เอกสาร")


if __name__ == "__main__":
    main()