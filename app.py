
import streamlit as st
import os
import pickle
from fastembed import TextEmbedding
import chromadb
from groq import Groq

st.set_page_config(page_title="UI Postgraduate AI Assistant")
st.title("University of Ibadan Postgraduate AI Assistant")
st.write("Ask a question about postgraduate admissions, thesis guidelines, examinations, or academic regulations.")

@st.cache_resource
def load_pipeline():
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client_db = chromadb.PersistentClient(path="chroma_db")
    collection = client_db.get_or_create_collection(name="ui_pg_knowledge_base")

    groq_client = Groq(api_key="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")

    return model, collection, groq_client

model, collection, groq_client = load_pipeline()

def search(query, n_results=4):
    query_embedding = list(model.embed([query]))[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results

def answer_question(question, n_results=4):
    results = search(question, n_results=n_results)
    retrieved_chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]

    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src in zip(retrieved_chunks, sources)
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
    return {"answer": answer, "sources": list(set(sources))}

question = st.text_input("Type your question here:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = answer_question(question)
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Sources")
                st.write(", ".join(result["sources"]))
            except Exception as e:
                st.error("Something went wrong reaching the AI service. Please check your internet connection and try again.")
