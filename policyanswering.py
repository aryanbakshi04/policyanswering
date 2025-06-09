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
    # The pagination block on Sansad uses <ul class="pagination">
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
def fetch_all_items(max_pages):
    items = []
    seen_pdfs = set()

    for page in range(1, max_pages + 1):
        url = f"{BASE_LIST_URL}?page={page}"
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # **Updated selector**: the Q&A rows live in a standard table
        rows = soup.select("table.table-striped tbody tr")
        st.write(f"Page {page}: found {len(rows)} rows")  # debug

        if not rows:
            break

        for row in rows:
            # 5th column holds the detail link
            link_cell = row.select_one("td:nth-of-type(5) a[href*='/ls/question/']")
            if not link_cell:
                continue

            detail_url = urljoin(BASE_LIST_URL, link_cell["href"])
            dresp = requests.get(detail_url)
            dresp.raise_for_status()
            dsoup = BeautifulSoup(dresp.text, "html.parser")

            pdf_tag = dsoup.select_one("a[href$='.pdf']")
            if not pdf_tag:
                continue

            pdf_url = urljoin(detail_url, pdf_tag["href"])
            if pdf_url in seen_pdfs:
                continue
            seen_pdfs.add(pdf_url)

            ministry   = dsoup.find("th", text="Ministry")   .find_next_sibling("td").get_text(strip=True)
            session    = dsoup.find("th", text="Session")    .find_next_sibling("td").get_text(strip=True)
            answer_date= dsoup.find("th", text="Answer Date").find_next_sibling("td").get_text(strip=True)

            items.append({
                "pdf_url": pdf_url,
                "ministry": ministry,
                "session": session,
                "answer_date": answer_date
            })

        st.write(f"Accumulated {len(items)} PDFs so far")  # debug

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

st.set_page_config(page_title="Lok Sabha QA Assistant", layout="wide")
st.title("Lok Sabha Q&A Assistant")

# 1. Detect total pages
total_pages = detect_total_pages()
st.sidebar.write(f"Detected {total_pages} pages of Q&A listings")

# 2. Fetch & index everything up front
items   = fetch_all_items(total_pages)
vectordb= build_vectordb(items)
if vectordb is None:
    st.error("Failed to index any documents. Please verify selectors or site structure.")
    st.stop()

agent   = init_agent()

# 3. Ministry dropdown (all ministries seen in the crawl)
ministries = sorted({it["ministry"] for it in items})
selected_min = st.sidebar.selectbox("Select Ministry", ["All"] + ministries)

# 4. User input
question = st.text_area("Enter your parliamentary question:")
if st.button("Get Answer"):
    if not question:
        st.error("cannot answer this query")
        st.stop()

    # 5. RAG retrieval
    docs = vectordb.similarity_search(question, k=10)
    if selected_min != "All":
        docs = [d for d in docs if d.metadata["ministry"] == selected_min]
    docs = docs[:4]

    if not docs:
        st.error("cannot answer this query")
    else:
        # 6. Build prompt & run Gemini
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

        # 7. Display
        st.subheader("📝 Answer")
        st.write(answer)
        st.subheader("📄 Sources")
        for d in docs:
            md = d.metadata
            st.markdown(f"- Session: {md['session']} | Date: {md['answer_date']}  ")
            st.markdown(f"  [PDF Source]({md['pdf_url']})")
