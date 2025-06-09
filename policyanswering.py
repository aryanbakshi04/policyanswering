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

BASE_LIST_URL = "https://sansad.in/ls/questions/questions-and-answers"
PDF_CACHE_DIR = "pdf_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

@st.cache_data
def fetch_all_items(max_pages=625):
    items = []
    for page in range(1, max_pages):
        resp = requests.get(f"{BASE_LIST_URL}?page={page}")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.select("table.table-striped tbody tr"):
            link_tag = row.select_one("td:nth-of-type(5) a")
            if not link_tag:
                continue
            detail_url = urljoin(BASE_LIST_URL, link_tag["href"])
            dresp = requests.get(detail_url)
            dresp.raise_for_status()
            dsoup = BeautifulSoup(dresp.text, "html.parser")
            ministry = dsoup.find("th", text="Ministry").find_next_sibling("td").get_text(strip=True)
            session = dsoup.find("th", text="Session").find_next_sibling("td").get_text(strip=True)
            ans_date = dsoup.find("th", text="ANSWERED ON").find_next_sibling("td").get_text(strip=True)
            pdf_tag = dsoup.select_one("a[href$='.pdf']")
            if not pdf_tag:
                continue
            pdf_url = urljoin(detail_url, pdf_tag["href"])
            items.append({
                "detail_url": detail_url,
                "ministry": ministry,
                "session": session,
                "answer_date": ans_date,
                "pdf_url": pdf_url
            })
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
    docs = []
    for it in items:
        pdf_path = download_pdf(it["pdf_url"])
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for p in pages:
            p.metadata.update({
                "ministry": it["ministry"],
                "session": it["session"],
                "answer_date": it["answer_date"],
                "source_pdf": it["pdf_url"]
            })
            docs.append(p)
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Answer as the ministry on Lok Sabha questions, using provided context.",
        instructions=[
            "Use the retrieved PDF context.",
            "Provide a formal, solution-oriented answer from the specified ministry."
        ],
        show_tool_calls=False,
        markdown=False
    )

st.set_page_config(page_title="Lok Sabha QA Assistant", layout="wide")
st.title("Lok Sabha Q&A Assistant (RAG + Gemini)")

max_pages = st.sidebar.slider("Pages of Q&A to index", 1, 10, 3)

if "vectordb" not in st.session_state:
    with st.spinner("Gathering & indexing Q&A PDFs..."):
        items = fetch_all_items(max_pages)
        vectordb = build_vectordb(items)
        if vectordb is None:
            st.error("No documents indexed. Check site structure or pagination.")
            st.stop()
        ministries = sorted({it["ministry"] for it in items})
        st.session_state.items = items
        st.session_state.ministries = ministries
        st.session_state.vectordb = vectordb
        st.session_state.agent = init_agent()

question = st.text_area("Enter your parliamentary question text")
selected_ministry = st.sidebar.selectbox(
    "Filter by Ministry", ["All"] + st.session_state.ministries
)

if st.button("Get Answer"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        docs = st.session_state.vectordb.similarity_search(question, k=10)
        if selected_ministry != "All":
            filtered = [d for d in docs if d.metadata.get("ministry") == selected_ministry]
            docs = filtered or docs[:4]
        else:
            docs = docs[:4]
        if not docs:
            st.error("No relevant context found for this ministry.")
        else:
            context = "\n\n".join(
                f"[{d.metadata['session']} | {d.metadata['answer_date']}] {d.page_content.strip()}"
                for d in docs
            )
            prompt = (
                f"Context:\n{context}\n\n"
                f"Answer as the {selected_ministry} ministry: Provide a formal, solution-oriented response. "
                f"Include Lok Sabha session, answer date, and source PDF link.\nQuestion: {question}"
            )
            with st.spinner("Generating answer..."):
                response = st.session_state.agent.run(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Sources")
            for d in docs:
                md = d.metadata
                st.markdown(
                    f"- Session: {md['session']}  \n"
                    f"  Date: {md['answer_date']}  \n"
                    f"  [PDF Source]({md['source_pdf']})"
                )
