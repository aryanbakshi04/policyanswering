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
    nums = [int(a.text) for a in pager.select("a") if a.text.isdigit()]
    return max(nums) if nums else 1

@st.cache_data
def fetch_all_items(max_pages):
    items, seen = [], set()
    for page in range(1, max_pages + 1):
        resp = requests.get(f"{BASE_LIST_URL}?page={page}")
        resp.raise_for_status()
        rows = BeautifulSoup(resp.text, "html.parser")\
               .select("table.table-striped tbody tr")
        if not rows:
            break
        for row in rows:
            link = row.select_one("td:nth-of-type(5) a")
            if not link:
                continue
            detail_url = urljoin(BASE_LIST_URL, link["href"])
            dresp = requests.get(detail_url); dresp.raise_for_status()
            dsoup = BeautifulSoup(dresp.text, "html.parser")
            pdf_tag = dsoup.select_one("a[href$='.pdf']")
            if not pdf_tag:
                continue
            pdf_url = urljoin(detail_url, pdf_tag["href"])
            if pdf_url in seen:
                continue
            seen.add(pdf_url)
            ministry = dsoup.find("th", text="Ministry")\
                          .find_next_sibling("td").text.strip()
            session  = dsoup.find("th", text="Session")\
                          .find_next_sibling("td").text.strip()
            date     = dsoup.find("th", text="Answer Date")\
                          .find_next_sibling("td").text.strip()
            items.append({
                "pdf_url": pdf_url,
                "ministry": ministry,
                "session": session,
                "answer_date": date
            })
    return items

@st.cache_data
def download_pdf(pdf_url):
    fname = os.path.basename(pdf_url)
    path = os.path.join(PDF_CACHE_DIR, fname)
    if not os.path.exists(path):
        r = requests.get(pdf_url); r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path

@st.cache_resource
def build_vectordb(items):
    if not items:
        return None
    docs = []
    for it in items:
        path = download_pdf(it["pdf_url"])
        for page in PyPDFLoader(path).load():
            page.metadata.update(it)
            docs.append(page)
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)\
             .split_documents(docs)
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
            "Use provided PDF context from past Q&As.",
            "Be formal, solution-oriented, positive, and focus on public welfare."
        ],
        show_tool_calls=False,
        markdown=False
    )

st.set_page_config(page_title="Lok Sabha QA Assistant", layout="wide")
st.title("Lok Sabha Q&A Assistant")

total_pages = detect_total_pages()
items = fetch_all_items(total_pages)
vectordb = build_vectordb(items)
if vectordb is None:
    st.error("Failed to index any documents. Cannot answer queries.")
    st.stop()
    
agent   = init_agent()

ministries = sorted({it["ministry"] for it in items})
selected_min = st.selectbox("Select Ministry", ["All"] + ministries)

question = st.text_area("Enter your parliamentary question:")
if st.button("Get the Response"):
    if not question.strip():
        st.error("cannot answer this query")
        st.stop()
    else:
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
            f"Answer as the Ministry of {selected_min if selected_min!='All' else 'your selection'}: "
            "Provide a formal, solution-oriented response focused on public interest. "
            "Include Lok Sabha session, answer date, and source PDF link.\n"
            f"Question: {question}"
        )
        response = agent.run(prompt)
        answer = getattr(response, "content", str(response))
        st.subheader("📝 Answer")
        st.write(answer)

        st.subheader("📄 Sources")
        for d in docs:
            md = d.metadata
            st.markdown(f"- Session: {md['session']} | Date: {md['answer_date']}")
            st.markdown(f"  [PDF Source]({md['pdf_url']})")
