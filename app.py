import html
import json
import os
import re
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Settings you may edit
# ---------------------------------------------------------------------------
GROUP_NAME = "Group 5"
GROUP_MEMBERS = [
    "Fagbenjo Samson Akinyele",
    "Ayantunmbi Raphael Ayantayo",
    "Alagbe Olorunniisola",
    "Samson Ariyo-Olaitan",
    "Ilyas Khadijah",
    "Noble Azoribe Ifechukwu",
    "Adigun Oluwaseunayo Elijah",
    "Oloyede Paul Promise",
]

SUGGESTED_QUESTIONS = [
    "What are the general admission requirements for postgraduate programmes?",
    "What is the required format for a thesis?",
    "How do I apply for the Teaching and Research Assistantship?",
    "Who needs to approve a request to conduct an oral examination?",
]

LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "ui_pg_knowledge_base"
N_RESULTS = 4          # chunks retrieved per question (same as the notebook)
HISTORY_MESSAGES = 8   # how many earlier messages the model can see
DB_PATH = os.environ.get("CHAT_DB_PATH", "chat_history.db")

SYSTEM_PROMPT = """You are the University of Ibadan Postgraduate Assistant, a friendly and helpful guide for students.
You answer using ONLY the numbered passages from official university documents that come with the student's message.

How to reply:
1. Be warm, natural and conversational, like a helpful senior colleague. Use plain language and short paragraphs. Use a short list only when it really helps. A greeting or a thank you gets a short, friendly reply.
2. Take facts only from the passages. Never invent rules, fees, dates, procedures, forms or names.
3. If the passages do not answer the question, say so kindly in your own words. Mention any related information you did find, suggest how the student could rephrase or what else you can help with, and say the Postgraduate College can confirm. Do not repeat one fixed sentence.
4. Use the earlier conversation to understand follow-up questions, and to answer questions about the chat itself, such as what the student has asked so far. For those you do not need the passages.
5. Do not write passage numbers or the words "passage" or "context" in your answer.
6. Finish with one last line that starts with SOURCES: followed by the numbers of the passages you actually used, for example SOURCES: 1, 3. If you used none (a greeting, a question about the chat, or passages that did not help), write SOURCES: none."""

st.set_page_config(
    page_title="UI Postgraduate Assistant",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
:root { --navy:#13284B; --navy2:#1C3A6B; --gold:#C9A227; --gold-soft:#E7D7A3; --paper:#F7F5F0; --slate:#2B2B2B; --muted:#64708A; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--slate); }
.stApp { background-color:var(--paper); }
section[data-testid="stSidebar"] { background-color:var(--navy); }
section[data-testid="stSidebar"] * { color:#FFFFFF !important; }
section[data-testid="stSidebar"] .stButton > button {
    background-color:var(--navy2); border:1px solid rgba(255,255,255,0.15);
    border-radius:8px; text-align:left; width:100%; justify-content:flex-start;
}
section[data-testid="stSidebar"] .stButton > button:hover { border-color:var(--gold); background-color:#23437E; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    border-color:var(--gold); background-color:#23437E;
}
.brand { display:flex; align-items:center; gap:12px; padding:4px 0 16px 0; border-bottom:1px solid rgba(255,255,255,0.15); margin-bottom:16px; }
.crest { width:40px; height:40px; border-radius:50%; background:var(--gold); display:flex; align-items:center;
         justify-content:center; font-family:'Lora',serif; font-weight:700; color:var(--navy) !important; flex-shrink:0; }
.brand-title { font-family:'Lora',serif; font-size:16px; font-weight:600; line-height:1.25; }
.brand-sub { font-size:12px; color:var(--gold-soft) !important; }
.statbox { background:var(--navy2); border-radius:10px; padding:10px 14px; margin:14px 0; font-size:13px; line-height:1.6; }
.main-title { font-family:'Lora',serif; font-size:28px; font-weight:700; color:var(--navy); margin-bottom:2px; }
.main-sub { color:var(--muted); font-size:14.5px; margin-bottom:20px; }
[class*="st-key-sugg_"] button { border:1px solid var(--gold); background:#FFFFFF; color:var(--navy); border-radius:14px;
    text-align:left; height:100%; min-height:72px; font-size:14px; }
[class*="st-key-sugg_"] button:hover { background:var(--gold-soft); border-color:var(--gold); }
[data-testid="stChatMessage"] { background:#FFFFFF; border:1px solid #E7E3D8; border-radius:14px; padding:12px 16px; margin-bottom:10px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#EAEFF8; border-color:#D5DDEE; }
.sources-line { font-size:12.5px; color:var(--muted); border-top:1px solid #E4E0D6; margin-top:10px; padding-top:8px; }
.notice { font-size:12px; color:var(--muted); margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat storage (SQLite). Every visitor only sees their own chats.
# ---------------------------------------------------------------------------
_db_lock = threading.Lock()


@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, owner TEXT, title TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, "
                 "role TEXT, content TEXT, sources TEXT, created_at TEXT)")
    conn.commit()
    return conn


def db_list_chats(owner):
    with _db_lock:
        return get_db().execute(
            "SELECT id, title FROM chats WHERE owner = ? ORDER BY created_at DESC LIMIT 40", (owner,)
        ).fetchall()


def db_create_chat(owner):
    chat_id = uuid.uuid4().hex
    with _db_lock:
        get_db().execute("INSERT INTO chats (id, owner, title, created_at) VALUES (?, ?, ?, ?)",
                         (chat_id, owner, "New chat", datetime.now().isoformat()))
        get_db().commit()
    return chat_id


def db_chat_exists(owner, chat_id):
    with _db_lock:
        row = get_db().execute("SELECT 1 FROM chats WHERE id = ? AND owner = ?", (chat_id, owner)).fetchone()
    return row is not None


def db_set_title_if_new(chat_id, question):
    title = (question[:40] + "...") if len(question) > 40 else question
    with _db_lock:
        get_db().execute("UPDATE chats SET title = ? WHERE id = ? AND title = 'New chat'", (title, chat_id))
        get_db().commit()


def db_load_messages(chat_id):
    with _db_lock:
        rows = get_db().execute(
            "SELECT role, content, sources FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)
        ).fetchall()
    out = []
    for role, content, sources in rows:
        try:
            src = json.loads(sources) if sources else []
        except Exception:
            src = []
        out.append({"role": role, "content": content, "sources": src})
    return out


def db_save_message(chat_id, role, content, sources=None):
    with _db_lock:
        get_db().execute(
            "INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, json.dumps(sources or []), datetime.now().isoformat()),
        )
        get_db().commit()


# ---------------------------------------------------------------------------
# Knowledge base, embedding model and Groq client (loaded once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_pipeline():
    state = {"embedder": None, "collection": None, "groq": None, "status": {}, "docs": [], "chunks": 0}
    status = state["status"]

    vec = None
    try:
        from fastembed import TextEmbedding
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        vec = list(embedder.embed(["test"]))[0]
        state["embedder"] = embedder
        status["Embedding model"] = f"ready ({len(vec)} numbers per text)"
    except Exception as e:
        traceback.print_exc()
        status["Embedding model"] = f"ERROR {type(e).__name__}: {e}"

    try:
        import chromadb
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection(COLLECTION_NAME)
        items = collection.get(include=["metadatas"])
        docs = sorted({m["source"] for m in items["metadatas"]})
        state["collection"], state["docs"], state["chunks"] = collection, docs, collection.count()
        status["Knowledge base"] = f"ready ({len(docs)} documents, {state['chunks']} chunks)"
    except Exception as e:
        traceback.print_exc()
        status["Knowledge base"] = f"ERROR {type(e).__name__}: {e}"

    if state["embedder"] is not None and state["collection"] is not None:
        try:
            state["collection"].query(query_embeddings=[vec.tolist()], n_results=1)
            status["Search test"] = "ok"
        except Exception as e:
            traceback.print_exc()
            status["Search test"] = f"ERROR {type(e).__name__}: {e}"

    key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'").strip()
    if not key:
        status["Groq key"] = "MISSING (add GROQ_API_KEY under Environment on Render)"
    else:
        try:
            from groq import Groq
            state["groq"] = Groq(api_key=key)
            status["Groq key"] = "found"
        except Exception as e:
            traceback.print_exc()
            status["Groq key"] = f"ERROR {type(e).__name__}: {e}"
    return state


PIPE = load_pipeline()

# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
FOLLOW_UP_START = re.compile(r"^(and|also|then|so|but|what about|how about|why|what if|which|who|when|where)\b", re.I)
PRONOUNS = re.compile(r"\b(it|its|that|this|they|them|those|these|he|she|there)\b", re.I)


def search_query(question, history):
    """Short follow-ups such as 'and for PhD?' borrow the previous question so retrieval still finds the right pages."""
    previous = [m["content"] for m in history if m["role"] == "user"]
    words = question.split()
    if previous and len(words) <= 7 and (len(words) <= 4 or FOLLOW_UP_START.match(question.strip()) or PRONOUNS.search(question)):
        return previous[-1] + " " + question
    return question


def retrieve(query):
    vec = list(PIPE["embedder"].embed([query]))[0].tolist()
    res = PIPE["collection"].query(query_embeddings=[vec], n_results=N_RESULTS)
    return [{"text": d, "source": m["source"], "page": m["page"]} for d, m in zip(res["documents"][0], res["metadatas"][0])]


def build_messages(question, chunks, history):
    passages = "\n\n".join(f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-HISTORY_MESSAGES:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"Passages:\n{passages}\n\nStudent's message: {question}"})
    return messages


def open_stream(messages):
    kwargs = dict(model=LLM_MODEL, messages=messages, temperature=0.3, stream=True)
    try:
        return PIPE["groq"].chat.completions.create(reasoning_effort="low", **kwargs)
    except Exception as e:
        text = str(e).lower()
        if "reasoning" in text or "unsupported" in text or "unknown" in text:
            return PIPE["groq"].chat.completions.create(**kwargs)
        raise


def text_pieces(stream):
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece


SOURCES_LINE = re.compile(r"(?:\n|^)[ \t]*\**[ \t]*SOURCES[ \t]*:[ \t]*([^\n]*?)[ \t]*\**[ \t]*$", re.I)
TAIL = 70


def hide_sources_line(pieces, holder):
    """Streams the answer but keeps back the last few characters so the SOURCES line is never shown to the user."""
    buffer = ""
    for piece in pieces:
        buffer += piece
        if len(buffer) > TAIL:
            yield buffer[:-TAIL]
            buffer = buffer[-TAIL:]
    match = SOURCES_LINE.search(buffer)
    if match:
        holder["sources_raw"] = match.group(1)
        buffer = buffer[: match.start()]
    yield buffer.rstrip()


def pick_sources(chunks, raw):
    if raw is None:                       # the model forgot the line: fall back to everything retrieved
        picked = chunks
    elif re.search(r"\bnone\b", raw, re.I) or not re.findall(r"\d+", raw):
        return []
    else:
        picked = [chunks[int(n) - 1] for n in re.findall(r"\d+", raw) if 1 <= int(n) <= len(chunks)]
    seen, out = set(), []
    for c in picked:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            out.append({"source": c["source"], "page": c["page"], "text": c["text"][:700]})
    return out


def render_sources(sources):
    if not sources:
        return
    line = "; ".join(f"{s['source']} (page {s['page']})" for s in sources)
    st.markdown(f'<div class="sources-line"><b>Sources:</b> {html.escape(line)}</div>', unsafe_allow_html=True)
    with st.expander("View the passages used"):
        for s in sources:
            st.markdown(f"**{s['source']}, page {s['page']}**")
            st.write(s["text"])


# ---------------------------------------------------------------------------
# Who is using the app: each visitor gets a private id kept in the page link
# ---------------------------------------------------------------------------
if "uid" not in st.session_state:
    uid = st.query_params.get("u")
    if not uid:
        uid = uuid.uuid4().hex[:12]
        st.query_params["u"] = uid
    st.session_state.uid = uid
UID = st.session_state.uid

if "chat_id" not in st.session_state or not db_chat_exists(UID, st.session_state.chat_id):
    existing = db_list_chats(UID)
    st.session_state.chat_id = existing[0][0] if existing else db_create_chat(UID)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div class="brand"><div class="crest">UI</div>
    <div><div class="brand-title">UI Postgraduate<br/>Assistant</div><div class="brand-sub">{html.escape(GROUP_NAME)}</div></div></div>
    """, unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True, icon=":material/add:"):
        st.session_state.chat_id = db_create_chat(UID)
        st.rerun()

    st.caption("YOUR CHATS")
    for cid, title in db_list_chats(UID):
        is_current = cid == st.session_state.chat_id
        if st.button(title or "New chat", key=f"chat_{cid}", use_container_width=True,
                     type="primary" if is_current else "secondary"):
            st.session_state.chat_id = cid
            st.rerun()
    st.caption("Chats are saved under this page link. Bookmark the page to come back to them.")

    st.markdown(f"""
    <div class="statbox">{len(PIPE['docs'])} source documents<br/>{PIPE['chunks']:,} searchable chunks</div>
    """, unsafe_allow_html=True)

    with st.expander("Documents used"):
        for name in PIPE["docs"]:
            st.markdown(f"- {name}")
    with st.expander("Group members"):
        for member in GROUP_MEMBERS:
            st.markdown(f"- {member}")
    with st.expander("System status"):
        for label, value in PIPE["status"].items():
            st.markdown(f"**{label}:** {value}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">University of Ibadan Postgraduate Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Ask about admissions, thesis rules, forms, examinations, scholarships and the academic calendar. '
            'Answers come from official university documents.</div>', unsafe_allow_html=True)

history = db_load_messages(st.session_state.chat_id)

if not history:
    st.markdown("**You can start with one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            st.button(q, key=f"sugg_{i}", use_container_width=True,
                      on_click=lambda q=q: st.session_state.update(pending=q))

for m in history:
    with st.chat_message(m["role"], avatar=":material/school:" if m["role"] == "assistant" else ":material/person:"):
        st.write(m["content"])
        if m["role"] == "assistant":
            render_sources(m["sources"])

typed = st.chat_input("Type your question here...")
question = st.session_state.pop("pending", None) or typed

if question:
    problems = [f"{k}: {v}" for k, v in PIPE["status"].items() if v.startswith(("ERROR", "MISSING"))]
    with st.chat_message("user", avatar=":material/person:"):
        st.write(question)
    with st.chat_message("assistant", avatar=":material/school:"):
        if problems:
            st.error("The assistant is not ready yet. Details:\n\n" + "\n\n".join(problems))
        else:
            try:
                with st.spinner("Searching the documents..."):
                    chunks = retrieve(search_query(question, history))
                    stream = open_stream(build_messages(question, chunks, history))
                holder = {}
                answer = st.write_stream(hide_sources_line(text_pieces(stream), holder))
                answer = (answer or "").strip()
                if not answer:
                    raise RuntimeError("The language model returned an empty answer.")
                sources = pick_sources(chunks, holder.get("sources_raw"))
                render_sources(sources)
                db_save_message(st.session_state.chat_id, "user", question)
                db_save_message(st.session_state.chat_id, "assistant", answer, sources)
                if not history:
                    db_set_title_if_new(st.session_state.chat_id, question)
                    st.rerun()
            except Exception as e:
                traceback.print_exc()
                st.error("Sorry, I could not get an answer just now. Please try again in a moment.")
                with st.expander("Technical details"):
                    st.code(f"{type(e).__name__}: {e}")

st.markdown('<div class="notice">Answers are generated from university documents and may be incomplete. '
            'Please confirm important decisions with the Postgraduate College.</div>', unsafe_allow_html=True)
