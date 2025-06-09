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
import json
from urllib.parse import urlencode

from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration ---
PDF_CACHE_DIR = "pdf_cache_sansad"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
API_URL = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)
# ---------------------------------------------------------------------
@st.cache_data(ttl=24*3600)
def fetch_all_questions(lokNo=18, sessionNo=4, max_pages=625, page_size=10, locale="en"):
    all_questions = []
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for page in range(1, max_pages + 1):
        params = {
            "loksabhaNo": lokNo,
            "sessionNumber": sessionNo,
            "pageNo": page,
            "locale": locale,
            "pageSize": page_size
        }

        try:
            resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            
            # Parse the response
            data = resp.json()
            
            # Debug first page
            if page == 1:
                st.write("First page data structure:", type(data))
                st.write("First page data length:", len(data) if isinstance(data, list) else 0)

            # Process data - we know it's a list of dicts with 'listOfQuestions'
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "listOfQuestions" in item:
                        questions_list = item["listOfQuestions"]
                        if isinstance(questions_list, list):
                            for q in questions_list:
                                if isinstance(q, dict):
                                    ministry = q.get("ministry")
                                    if ministry:  # Only process if ministry exists
                                        processed_q = {
                                            "question_no": q.get("quesNo"),
                                            "subject": q.get("subjects"),
                                            "loksabha": q.get("lokNo"),
                                            "session": q.get("sessionNo"),
                                            "member": (", ".join(q.get("member", []))
                                                     if isinstance(q.get("member"), list)
                                                     else q.get("member", "")),
                                            "ministry": ministry,
                                            "type": q.get("type"),
                                            "pdf_url": q.get("questionsFilePath"),
                                            "question_text": q.get("questionText"),
                                            "date": q.get("date"),
                                        }
                                        all_questions.append(processed_q)

            # Debug counts
            if page == 1:
                st.write(f"Processed questions on page 1: {len(all_questions)}")
                if all_questions:
                    st.write("First processed question:", all_questions[0])

            # If no questions found on a page, assume we've reached the end
            if not any(isinstance(item, dict) and item.get("listOfQuestions") for item in data):
                break

        except Exception as e:
            st.error(f"Error processing page {page}: {str(e)}")
            import traceback
            st.write("Traceback:", traceback.format_exc())
            break

    # Final processing
    if all_questions:
        ministries = sorted({q['ministry'] for q in all_questions if q.get('ministry')})
        st.write(f"Total questions processed: {len(all_questions)}")
        st.write(f"Unique ministries found: {len(ministries)}")
        st.write("Ministries:", ministries)
    else:
        st.error("No questions were processed successfully")

    return all_questions

# --- Streamlit App UI ---
st.title("Parliamentary Ministry Q&A Assistant")

# Initialize session state
if 'all_records' not in st.session_state:
    st.session_state.all_records = None

# Fetch data
with st.spinner("Fetching parliamentary questions..."):
    all_records = fetch_all_questions()
    st.session_state.all_records = all_records

# Process ministries
if st.session_state.all_records:
    ministries = sorted({
        rec['ministry'] 
        for rec in st.session_state.all_records 
        if rec.get('ministry')
    })
    
    if ministries:
        st.write(f"Found {len(ministries)} ministries")
        selected_ministry = st.sidebar.selectbox("Select Ministry", ministries)
        
        # Filter records
        filtered = [r for r in st.session_state.all_records if r['ministry'] == selected_ministry]
        if filtered:
            st.write(f"Found {len(filtered)} questions for {selected_ministry}")
            
            # Continue with your vector store and Q&A logic...
        else:
            st.error(f"No records found for {selected_ministry}")
    else:
        st.error("No ministries found in the processed data")
else:
    st.error("No data retrieved from API")
# -------------------------------------------------------------------------------------
all_records = fetch_all_questions()
ministries = sorted({rec['ministry'] for rec in all_records if rec['ministry']})
st.write("Ministries found:", ministries)

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
# Add the debug print right after fetching
ministries = sorted({rec['ministry'] for rec in all_records if rec['ministry']})
st.write("Ministries found:", ministries)

if not ministries:
    st.error("No ministries found in fetched records. Check your API or extraction logic.")
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
