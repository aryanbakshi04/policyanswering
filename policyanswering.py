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

BASE_URL = "https://sansad.in/ls/questions/questions-and-answers"
PDF_CACHE_DIR = "pdf_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

def detect_total_pages():
    resp = requests.get(BASE_URL)
    soup = BeautifulSoup(resp.text, "html.parser")
    pager = soup.select_one("ul.pagination")
    if not pager:
        return 1
    return max([int(a.text) for a in pager.select("a") if a.text.isdigit()] or [1])

@st.cache_data
def fetch_all_items():
    items = []
    seen_pdfs = set()
    total_pages = detect_total_pages()
    for page in range(1, total_pages + 1):
        page_url = f"{BASE_URL}?page={page}"
        resp = requests.get(page_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.select("table tbody tr"):
            cols = row.find_all("td")
            if len(cols) < 5:
                continue
            ministry = cols[1].text.strip()
            session = cols[0].text.strip()
            date = cols[2].text.strip()
            pdf_link = cols[4].find("a", href=True)
            if not pdf_link:
                continue
            pdf_url = pdf_link["href"]
            if pdf_url in seen_pdfs:
                continue
            seen_pdfs.add(pdf_url)
            if not pdf_url.startswith("http"):
                pdf_url = "https://sansad.in" + pdf_url
            items.append({
                "ministry": ministry,
                "session": session,
                "answer_date": date,
                "pdf_url": pdf_url
            })
    return items

@st.cache_data
def download_pdf(pdf_url):
    fname = os.path.basename(pdf_url)
    path = os.path.join(PDF_CACHE_DIR, fname)
    if not os.path.exists(path):
        r = requests.get(pdf_url)
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
    chunks = splitter.split_documents(docs)
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
    st.error("Failed to fetch any Q&A items.")
    st.stop()

vectordb = build_vectordb(items)
if vectordb is None:
    st.error("Failed to index documents.")
    st.stop()

agent = init_agent()

ministries = sorted({it["ministry"] for it in items})
selected_min = st.sidebar.selectbox("Select Ministry", ["All"] + ministries)
question = st.text_area("Enter your parliamentary question:")

if st.button("Get Answer"):
    if not question:
        st.error("Please enter a question.")
        st.stop()
    docs = vectordb.similarity_search(question, k=10)
    if selected_min != "All":
        docs = [d for d in docs if d.metadata["ministry"] == selected_min]
    docs = docs[:4]
    if not docs:
        st.error("No relevant documents found.")
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
            st.markdown(f"- Session: {md['session']} | Date: {md['answer_date']}")
            st.markdown(f"  [PDF Source]({md['pdf_url']})")
