try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration ---
BASE_URL = "https://sansad.in/ls/questions/questions-and-answers"
PDF_CACHE_DIR = "pdf_cache_sansad"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

# --- Fetch all Q&A records across ministries ---
# Remove @st.cache_data for debugging; re-enable after fixing issues
def fetch_all_records(max_pages=50):
    import urllib.parse
    records = []
    for page in range(max_pages):
        params = {'page': page}
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
        except Exception as e:
            st.error(f"Network error on page {page}: {e}")
            break
        if resp.status_code != 200:
            st.error(f"Failed to fetch page {page}: HTTP {resp.status_code}")
            st.write("Partial response content:", resp.text[:1000])
            break
        soup = BeautifulSoup(resp.text, 'html.parser')
        rows = soup.select('table.views-table tbody tr')
        if not rows:
            st.warning(f"No table rows found on page {page}. The website structure might have changed.")
            st.write("Partial page content for debugging:", resp.text[:1000])
            break
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 8:
                continue
            question = cells[1].get_text(strip=True)
            session = cells[3].get_text(strip=True)
            date = cells[4].get_text(strip=True)
            ministry = cells[5].get_text(strip=True)
            link_tag = cells[7].find('a', href=True)
            pdf_url = urllib.parse.urljoin(BASE_URL, link_tag['href']) if link_tag else None
            records.append({
                'question': question,
                'session': session,
                'date': date,
                'ministry': ministry,
                'pdf_url': pdf_url
            })
    st.write(f"Fetched {len(records)} records.")
    return records

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
all_records = fetch_all_records()
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
                return

            # Display
            st.subheader("📝 Answer")
            st.write(answer)
            st.subheader("📋 Details")
            st.write(f"**Session:** {docs[0].metadata.get('session', 'N/A')}  ")
            st.write(f"**Date:** {docs[0].metadata.get('date', 'N/A')}  ")
            st.subheader("📄 Source PDF(s)")
            for url in {d.metadata.get('source_url') for d in docs if d.metadata.get('source_url')}:
                st.markdown(f"- [PDF Link]({url})")
