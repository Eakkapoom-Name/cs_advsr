# 🎓 Course Advisor

Course Advisor is an AI assistant that helps Computer Science students discover and plan courses. It centralizes curriculum knowledge and answers planning questions so students can make confident, up‑to‑date decisions.

## Overview
- **What it is:** an AI helper that recommends and explains courses for Computer Science students.  
- **Why it’s needed:** students face fragmented, hard‑to‑find information across syllabi, study plans, and registration pages; and generic AI often misses curriculum‑specific, up‑to‑date details.  
- **How it works:** builds a structured table from the official curriculum, course descriptions, and study plan, and uses an LLM with `RAG`, `Function Calling`, and careful `Prompt Design` to retrieve the right facts and answer questions accurately.  

## Members & Roles
- **Eakkapoom Mapeng (686)** — `Prompt design` `debugging (backend)` `backend ↔ frontend integration`  
- **Komphon Burutsri (691)** — `RAG implementation` `function calls`  
- **Jakkarin Haisok (693)** — `UX design` `debugging (frontend)` `frontend ↔ backend integration`  
- **Natthanicha Sompao (705)** — `Data collection` `UI design`  

## Setup

### Clone/Download this repository to your device
```bash
git clone <this-repo-url>
# or download the ZIP and extract it
```

### Open your terminal and move to the project directory
```bash
cd <path-to-repo-folder>
```

### Run `setup.bash` and follow on‑screen instructions
```bash
bash setup.bash
```

### Activate the virtual environment
`If it’s already active, you can skip this step.`
```bash
source .venv/bin/activate
```

### Run the app with Streamlit
```bash
streamlit run app.py
```
