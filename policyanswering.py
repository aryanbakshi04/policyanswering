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


BASE_LIST_URL      = "https://sansad.in/ls/questions/questions-and-answers"
PDF_CACHE_DIR      = "pdf_cache"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME  = "gemini-2.0-flash-exp"

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
        try:
            p = int(a.get_text(strip=True))
            pages.append(p)
        except ValueError:
            continue
    return max(pages) if pages else 1

@st.cache_data(show_spinner=False)
def fetch_all_items(max_pages):
    items = []
    seen_pdfs = set()
    progress = st.progress(0)
    last_count = 0

    for page in range(1, max_pages + 1):
        resp = requests.get(f"{BASE_LIST_URL}?page={page}")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table.table-striped tbody tr")
        if not rows:
            break  

        for row in rows:
            link_td = row.select_one("td:nth-of-type(5) a")
            if not link_td:
                continue
            detail_url = urljoin(BASE_LIST_URL, link_td["href"])
            dresp = requests.get(detail_url)
            dresp.raise_for_status()
            dsoup = BeautifulSoup(dresp.text, "html.parser")

            ministry = (
                dsoup.find("th", text="Ministry")
                     .find_next_sibling("td")
                     .get_text(strip=True)
            )
            session = (
                dsoup.find("th", text="Session")
                     .find_next_sibling("td")
                     .get_text(strip=True)
            )
            ans_date = (
                dsoup.find("th", text="Answer Date")
                     .find_next_sibling("td")
                     .get_text(strip=True)
            )
            pdf_tag = dsoup.select_one("a[href$='.pdf']")
            if not pdf_tag:
                continue

            pdf_url = urljoin(detail_url, pdf_tag["href"])
            if pdf_url in seen_pdfs:
                continue

            seen_pdfs.add(pdf_url)
            items.append({
                "ministry": ministry,
                "session": session,
                "answer_date": ans_date,
                "pdf_url": pdf_url
            })

        
        progress.progress(page / max_pages)
        
        if len(items) == last_count:
            break
        last_count = len(items)

    progress.empty()
    return items

@st.cache_data(show_spinner=False)
def download_pdf(pdf_url):
    fname = os.path.basename(pdf_url)
    path = os.path.join(PDF_CACHE_DIR, fname)
    if not os.path.exists(path):
        r = requests.get(pdf_url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path

@st.cache_resource(show_spinner=False)
def build_vectordb(items):
    if not items:
        return None

    docs = []
    for it in items:
        path = download_pdf(it["pdf_url"])
        loader = PyPDFLoader(path)
        for page in loader.load():
            page.metadata.update(it)
            docs.append(page)

    if not docs:
        return None

    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks     = splitter.split_documents(docs)
    if not chunks:
        return None

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb   = FAISS.from_documents(chunks, embeddings)
    return vectordb

@st.cache_resource(show_spinner=False)
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Answer Lok Sabha questions as the specified ministry.",
        instructions=[
            "Use the provided context from retrieved PDF snippets.",
            "Respond formally, solution-oriented, include session, date, PDF."
        ],
        show_tool_calls=False,
        markdown=False
    )


st.set_page_config(page_title="Lok Sabha QA Assistant", layout="wide")
st.title("Lok Sabha Q&A Assistant (Robust RAG + Gemini)")


total_pages = detect_total_pages()
pages_to_crawl = st.sidebar.number_input(
    "Pages to crawl", min_value=1, max_value=total_pages, value=min(5, total_pages)
)

if "vectordb" not in st.session_state:
    with st.spinner("Indexing Lok Sabha Q&A PDFs…"):
        items = fetch_all_items(pages_to_crawl)
        vectordb = build_vectordb(items)
        if vectordb is None:
            st.error("No documents indexed—try raising the page count or check site structure.")
            st.stop()

        st.session_state.items   = items
        st.session_state.vectordb = vectordb
        st.session_state.agent    = init_agent()


ministries = sorted({it["ministry"] for it in st.session_state.items})
selected_ministry = st.sidebar.selectbox("Filter by Ministry", ["All"] + ministries)

question = st.text_area("Parliamentary Question")
if st.button("Get Answer"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        docs = st.session_state.vectordb.similarity_search(question, k=10)
        if selected_ministry != "All":
            docs = [d for d in docs if d.metadata["ministry"] == selected_ministry]
        docs = docs[:4]

        if not docs:
            st.error("No relevant context found.")
        else:
            context = "\n\n".join(
                f"[{d.metadata['session']} | {d.metadata['answer_date']}] {d.page_content.strip()}"
                for d in docs
            )
            prompt = (
                f"Context:\n{context}\n\n"
                f"Answer as {selected_ministry if selected_ministry!='All' else 'the requested ministry'}: "
                f"Formal, solution-oriented. Include session, date, PDF link.\n"
                f"Question: {question}"
            )
            with st.spinner("Generating answer…"):
                resp = st.session_state.agent.run(prompt)

            answer = getattr(resp, "content", str(resp))
            st.subheader("Response")
            st.write(answer)

            st.subheader("Sources")
            for d in docs:
                md = d.metadata
                st.markdown(
                    f"- **Session:** {md['session']}  \n"
                    f"  **Date:** {md['answer_date']}  \n"
                    f"  [PDF]({md['pdf_url']})"
                )
