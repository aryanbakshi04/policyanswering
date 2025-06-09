try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration ---
PDF_CACHE_DIR = "pdf_cache_sansad"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
API_URL = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

@st.cache_data(ttl=24*3600)
def fetch_all_questions(loksabha_no=18, session_no=4, max_pages=625, page_size=10, locale="en"):
    all_questions = []
    for page in range(1, max_pages + 1):     # <-- start from 1, not 0
        params = {
            "loksabhaNo": loksabha_no,
            "sessionNumber": session_no,
            "pageNo": page,
            "locale": locale,
            "pageSize": page_size
        }
        try:
            resp = requests.get(API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.warning(f"Error fetching page {page}: {e}")
            break

        # ---- robust handling ----
        if isinstance(data, dict):
            data_obj = data.get("data", {})
            if isinstance(data_obj, dict):
                questions = data_obj.get("questions", [])
            elif isinstance(data_obj, list):
                questions = data_obj
            else:
                questions = []
        elif isinstance(data, list):
            questions = data
        else:
            questions = []

        if not questions:
            break

        for q in questions:
            pdf_url = q.get("pdfPath")
            if pdf_url and not pdf_url.startswith("http"):
                pdf_url = "https://sansad.in" + pdf_url
            all_questions.append({
                "question_no": q.get("questionNumber"),
                "subject": q.get("subject"),
                "loksabha": q.get("loksabhaNo"),
                "session": q.get("sessionNumber"),
                "member": q.get("questionByMemberNames"),
                "ministry": q.get("ministry") or q.get("ministryName"),
                "type": q.get("questionType"),
                "pdf_url": pdf_url,
                "question_text": q.get("questionTitle"),
                "date": q.get("answerDate"),
            })
    return all_questions
# --- Build FAISS vector store from filtered records ---
@st.cache_resource
def build_vectorstore(records):
    docs = []
    for rec in records:
        if rec['pdf_url']:
            fname = os.path.join(PDF_CACHE_DIR, os.path.basename(rec['pdf_url']))
            if not os.path.exists(fname):
                try:
                    r = requests.get(rec['pdf_url'], timeout=30)
                    r.raise_for_status()
                    with open(fname, 'wb') as f: f.write(r.content)
                except Exception as e:
                    st.warning(f"Failed to download PDF: {rec['pdf_url']}\nError: {e}")
                    continue
            loader = PyPDFLoader(fname)
            try:
                loaded = loader.load()
            except Exception as e:
                st.warning(f"Failed to load PDF {fname}: {e}")
                continue
            for d in loaded:
                d.metadata.update({
                    'session': rec['session'],
                    'date': rec['date'],
                    'ministry': rec['ministry'],
                    'source_url': rec['pdf_url']
                })
            docs.extend(loaded)
    if not docs:
        st.error("No documents loaded for this ministry. Check for PDF download/parsing errors.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

# --- Initialize Gemini Agent ---
@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Answers parliamentary ministry questions based on retrieved context.",
        instructions=[
            "Use context from ministry Q&A PDFs.",
            "Provide formal, solution-oriented responses."
        ],
        show_tool_calls=False,
        markdown=False
    )

# --- Streamlit App UI ---
st.title("Parliamentary Ministry Q&A Assistant")

# Load and cache all records once
all_records = fetch_all_questions()
if not all_records:
    st.error("Unable to fetch Q&A records. Check site or network.")
    st.stop()

# Dropdown of ministries
ministries = sorted({rec['ministry'] for rec in all_records if rec['ministry']})
if not ministries:
    st.error("No ministries found in fetched records.")
    st.stop()

selected_ministry = st.sidebar.selectbox("Select Ministry", ministries)

# Filter records by ministry
filtered = [r for r in all_records if r['ministry'] == selected_ministry]
if not filtered:
    st.error("No records found for selected ministry.")
    st.stop()

# Build vector store for this ministry
vectordb = build_vectorstore(filtered)
if vectordb is None:
    st.error("No searchable documents available for this ministry.")
    st.stop()

agent = init_agent()

# User question input
question = st.text_area("Your Parliamentary Question:")
if st.button("Get Ministry Response"):
    if not question.strip():
        st.error("Cannot answer this query. Please enter a question.")
    else:
        docs = vectordb.similarity_search(question, k=5)
        if not docs:
            st.error("No relevant information found to answer this query.")
        else:
            context = "\n\n".join(d.page_content for d in docs)
            prompt = (
                f"Context:\n{context}\n\n"
                f"Provide an answer on behalf of {selected_ministry} ministry, including session and date. Question: {question}"
            )
            try:
                response = agent.run(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

            # Display
            st.subheader("📝 Answer")
            st.write(answer)
            st.subheader("📋 Details")
            st.write(f"**Session:** {docs[0].metadata.get('session', 'N/A')}  ")
            st.write(f"**Date:** {docs[0].metadata.get('date', 'N/A')}  ")
            st.subheader("📄 Source PDF(s)")
            for url in {d.metadata.get('source_url') for d in docs if d.metadata.get('source_url')}:
                st.markdown(f"- [PDF Link]({url})")
