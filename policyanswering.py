try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

BASE_LIST_URL     = "https://sansad.in/ls/questions/questions-and-answers"
PDF_CACHE_DIR     = "pdf_cache"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

def detect_total_pages():
    resp = requests.get(BASE_LIST_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    pager = soup.select_one("ul.pagination")
    if not pager:
        return 1
    pages = []
    for a in pager.find_all("a", href=True):
        text = a.get_text(strip=True)
        if text.isdigit():
            pages.append(int(text))
    return max(pages) if pages else 1

@st.cache_data
def fetch_all_items():
    items = []
    seen_pdfs = set()
    page = 1
    while True:
        api_url = (
            "https://sansad.in/api/question/getLSQpage"
            f"?page={page}&lang=en&type=LSQ"
        )
        resp = requests.get(api_url)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        if not rows:
            break
        for row in rows:
            ministry    = row.get("ministryName", "").strip()
            session     = row.get("lakshadSession", "").strip() or row.get("session", "").strip()
            q_date      = row.get("date", "").strip()
            pdf_url     = row.get("pdf", "").strip()
            if not pdf_url or pdf_url in seen_pdfs:
                continue
            seen_pdfs.add(pdf_url)
            items.append({
                "ministry":    ministry,
                "session":     session,
                "answer_date": q_date,
                "pdf_url":     pdf_url
            })
        page += 1
    return items

@st.cache_data
def download_pdf(pdf_url):
    fname = os.path.basename(pdf_url)
    path = os.path.join(PDF_CACHE_DIR, fname)
    if not os.path.exists(path):
        r = requests.get(pdf_url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path

@st.cache_resource
def build_vectordb(items):
    if not items:
        return None
    docs = []
    for it in items:
        pdf_path = download_pdf(it["pdf_url"])
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for p in pages:
            p.metadata.update(it)
            docs.append(p)
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks   = splitter.split_documents(docs)
    if not chunks:
        return None
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Answer Lok Sabha questions as the selected ministry.",
        instructions=[
            "Use provided PDF snippets as context.",
            "Be formal, positive, solution-oriented, and focus on public welfare."
        ],
        show_tool_calls=False,
        markdown=False
    )

items = fetch_all_items()
if not items:
    st.error("Failed to fetch any Q&A items. The API may have changed.")
    st.stop()

vectordb = build_vectordb(items)
if vectordb is None:
    st.error("Failed to index any documents. Cannot answer queries.")
    st.stop()
agent = init_agent()

ministries   = sorted({it["ministry"] for it in items})
selected_min = st.selectbox("Select Ministry", ["All"] + ministries)

question = st.text_area("Enter your parliamentary question:")
if st.button("Get Answer"):
    pass

total_pages = detect_total_pages()
st.sidebar.write(f"Detected {total_pages} pages of Q&A listings")

items   = fetch_all_items()
vectordb= build_vectordb(items)
if vectordb is None:
    st.error("Failed to index any documents. Please verify selectors or site structure.")
    st.stop()

agent   = init_agent()

ministries = sorted({it["ministry"] for it in items})
selected_min = st.sidebar.selectbox("Select Ministry", ["All"] + ministries)

question = st.text_area("Enter your parliamentary question:")
if st.button("Get Answer"):
    if not question:
        st.error("cannot answer this query")
        st.stop()
    docs = vectordb.similarity_search(question, k=10)
    if selected_min != "All":
        docs = [d for d in docs if d.metadata["ministry"] == selected_min]
    docs = docs[:4]
    if not docs:
        st.error("cannot answer this query")
    else:
        context = "\n\n".join(
            f"[{d.metadata['session']} | {d.metadata['answer_date']}] {d.page_content.strip()}"
            for d in docs
        )
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer as the Ministry of {selected_min}: Provide a formal, solution-oriented response "
            f"focusing on public interest. Include session, date, and source PDF link.\nQuestion: {question}"
        )
        response = agent.run(prompt)
        answer = getattr(response, "content", str(response))
        st.subheader("Response")
        st.write(answer)
        st.subheader("Sources:")
        for d in docs:
            md = d.metadata
            st.markdown(f"- Session: {md['session']} | Date: {md['answer_date']}  ")
            st.markdown(f"  [PDF Source]({md['pdf_url']})")
