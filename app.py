import os
import sqlite3
import uuid
from datetime import datetime

import streamlit as st
from fastembed import TextEmbedding
import chromadb
from groq import Groq

# ---------------------------------------------------------------------------
# Group info — edit these before deploying
# ---------------------------------------------------------------------------
GROUP_NAME = "Group Five"
GROUP_MEMBERS = [
    "Member 1 — Full Name",
    "Member 2 — Full Name",
    "Member 3 — Full Name",
    "Member 4 — Full Name",
    "Member 5 — Full Name",
    "Member 6 — Full Name",
    "Member 7 — Full Name",
]

SUGGESTED_QUESTIONS = [
    "What are the admission requirements for postgraduate programmes?",
    "What is the required format for a thesis?",
    "How do I apply for a PhD scholarship?",
    "What is the role of the Postgraduate College?",
]

DB_PATH = "chat_history.db"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UI Postgraduate Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink-navy: #13284B;
    --ink-navy-light: #1C3A6B;
    --gold: #C9A227;
    --gold-soft: #E7D7A3;
    --paper: #F7F5F0;
    --slate: #2B2B2B;
    --muted: #64708A;
    --white: #FFFFFF;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--slate); }

.stApp { background-color: var(--paper); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--ink-navy);
}
section[data-testid="stSidebar"] * { color: var(--white) !important; }
section[data-testid="stSidebar"] .stButton>button {
    background-color: var(--ink-navy-light);
    border: 1px solid rgba(255,255,255,0.15);
    color: var(--white) !important;
    border-radius: 8px;
    text-align: left;
    width: 100%;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    border-color: var(--gold);
    background-color: #23437E;
}

/* Brand header */
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 0 18px 0;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 18px;
}
.brand-crest {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: var(--gold);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Lora', serif; font-weight: 700; font-size: 18px;
    color: var(--ink-navy);
    flex-shrink: 0;
}
.brand-title { font-family: 'Lora', serif; font-size: 17px; font-weight: 600; line-height: 1.25; }
.brand-subtitle { font-size: 12px; color: var(--gold-soft) !important; }

/* Main header */
.main-header {
    font-family: 'Lora', serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--ink-navy);
    margin-bottom: 2px;
}
.main-subheader {
    color: var(--muted);
    font-size: 14.5px;
    margin-bottom: 22px;
}

/* Suggested question chips */
.stButton>button[kind="secondary"] {
    border-radius: 20px;
    border: 1px solid var(--gold);
    color: var(--ink-navy);
    background: var(--white);
    font-size: 13.5px;
    padding: 6px 14px;
}
.stButton>button[kind="secondary"]:hover {
    background: var(--gold-soft);
}

/* Chat bubbles */
[data-testid="stChatMessage"] { padding: 2px 0; }

/* Sources line */
.sources-line {
    font-size: 12px;
    color: var(--muted);
    border-top: 1px solid #E4E0D6;
    margin-top: 8px;
    padding-top: 6px;
}

/* Stats box */
.stats-box {
    background: var(--ink-navy-light);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Database (stores chat sessions and messages so history survives a reload)
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            created_at TEXT
        )
    """)
    return conn

db = get_db()

def list_chats():
    rows = db.execute("SELECT id, title FROM chats ORDER BY created_at DESC").fetchall()
    return rows

def create_chat():
    chat_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
        (chat_id, "New chat", datetime.now().isoformat()),
    )
    db.commit()
    return chat_id

def rename_chat_if_untitled(chat_id, first_question):
    row = db.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row and row[0] == "New chat":
        title = (first_question[:42] + "…") if len(first_question) > 42 else first_question
        db.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
        db.commit()

def load_messages(chat_id):
    rows = db.execute(
        "SELECT role, content, sources FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()
    return rows

def save_message(chat_id, role, content, sources=""):
    db.execute(
        "INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, role, content, sources, datetime.now().isoformat()),
    )
    db.commit()

# ---------------------------------------------------------------------------
# RAG pipeline (loaded once, reused across the whole session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the knowledge base…")
def load_pipeline():
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client_db = chromadb.PersistentClient(path="chroma_db")
    collection = client_db.get_or_create_collection(name="ui_pg_knowledge_base")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return model, collection, groq_client

model, collection, groq_client = load_pipeline()

@st.cache_data(show_spinner=False)
def get_knowledge_base_stats():
    all_items = collection.get(include=["metadatas"])
    sources = set(m["source"] for m in all_items["metadatas"])
    return len(sources), collection.count(), sorted(sources)

doc_count, chunk_count, doc_names = get_knowledge_base_stats()

def search(query, n_results=4):
    query_embedding = list(model.embed([query]))[0].tolist()
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)

def answer_question(question, n_results=4):
    results = search(question, n_results=n_results)
    retrieved_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source: {meta['source']}, page {meta['page']}]\n{chunk}"
        for chunk, meta in zip(retrieved_chunks, metadatas)
    )

    prompt = f"""You are an assistant helping students of the University of Ibadan Postgraduate College.
Answer the question using ONLY the information in the context below.
If the context does not contain enough information to answer, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    source_pages = sorted(set((m["source"], m["page"]) for m in metadatas))
    sources = [f"{src} (page {pg})" for src, pg in source_pages]
    return answer, sources

# ---------------------------------------------------------------------------
# Session state: which chat is currently open
# ---------------------------------------------------------------------------
if "current_chat_id" not in st.session_state:
    existing = list_chats()
    st.session_state.current_chat_id = existing[0][0] if existing else create_chat()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div class="brand-header">
        <div class="brand-crest">UI</div>
        <div>
            <div class="brand-title">UI Postgraduate<br/>Assistant</div>
            <div class="brand-subtitle">{GROUP_NAME}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.current_chat_id = create_chat()
        st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.caption("YOUR CHATS")

    for chat_id, title in list_chats():
        label = title if title else "New chat"
        if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stats-box">
        📚 {doc_count} source documents<br/>
        🧩 {chunk_count:,} searchable chunks
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Documents used"):
        for name in doc_names:
            st.markdown(f"- {name}")

    with st.expander("Group members"):
        for m in GROUP_MEMBERS:
            st.markdown(f"- {m}")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">University of Ibadan Postgraduate Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subheader">Ask about admissions, thesis guidelines, examinations, scholarships, and academic regulations — answered from official university documents.</div>', unsafe_allow_html=True)

messages = load_messages(st.session_state.current_chat_id)

if not messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    clicked_suggestion = None
    for col, q in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(q, key=f"sugg_{q}", use_container_width=True, type="secondary"):
                clicked_suggestion = q
else:
    clicked_suggestion = None

for role, content, sources in messages:
    with st.chat_message(role, avatar="🎓" if role == "assistant" else "🧑"):
        st.write(content)
        if sources:
            st.markdown(f'<div class="sources-line">Sources: {sources}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Type your question here…")
question_to_ask = user_input or clicked_suggestion

if question_to_ask:
    save_message(st.session_state.current_chat_id, "user", question_to_ask)
    rename_chat_if_untitled(st.session_state.current_chat_id, question_to_ask)

    with st.chat_message("user", avatar="🧑"):
        st.write(question_to_ask)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching documents and generating an answer…"):
            try:
                answer, sources = answer_question(question_to_ask)
                sources_line = ", ".join(sources)
            except Exception:
                answer = "Something went wrong reaching the AI service. Please check your connection and try again."
                sources_line = ""
        st.write(answer)
        if sources_line:
            st.markdown(f'<div class="sources-line">Sources: {sources_line}</div>', unsafe_allow_html=True)

    save_message(st.session_state.current_chat_id, "assistant", answer, sources_line)
    st.rerun()
