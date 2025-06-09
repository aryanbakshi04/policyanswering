try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os, streamlit as st, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

BASE_URL      = "https://sansad.in/ls/questions/questions-and-answers"
HEADERS       = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}
PDF_CACHE_DIR = "pdf_cache"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL  = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

def detect_total_pages():
    resp = requests.get(BASE_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    pager = soup.select_one("ul.pagination")
    if not pager:
        return 1
    nums = [int(a.text) for a in pager.select("a") if a.text.isdigit()]
    return max(nums) if nums else 1

@st.cache_data
def fetch_all_items():
    total = detect_total_pages()
    items, seen = [], set()
    for pg in range(1, total+1):
        listing = f"{BASE_URL}?page={pg}"
        r = requests.get(listing, headers=HEADERS)
        r.raise_for_status()
        rows = BeautifulSoup(r.text, "html.parser") \
               .select("table.table-striped tbody tr")
        if not rows:
            break
        for row in rows:
            link = row.select_one("td:nth-of-type(5) a")
            if not link: 
                continue
            detail_url = urljoin(BASE_URL, link["href"])
            dr = requests.get(detail_url, headers=HEADERS)
            dr.raise_for_status()
            ds = BeautifulSoup(dr.text, "html.parser")
            pdf_tag = ds.select_one("a[href$='.pdf']")
            if not pdf_tag:
                continue
            pdf_url = urljoin(detail_url, pdf_tag["href"])
            if pdf_url in seen:
                continue
            seen.add(pdf_url)
            ministry    = ds.find("th", text="Ministry").find_next_sibling("td").text.strip()
            session     = ds.find("th", text="Session").find_next_sibling("td").text.strip()
            answer_date = ds.find("th", text="Answer Date").find_next_sibling("td").text.strip()
            items.append({
                "pdf_url":     pdf_url,
                "ministry":    ministry,
                "session":     session,
                "answer_date": answer_date
            })
    return items

@st.cache_data
def download_pdf(url):
    fname = os.path.basename(url)
    path = os.path.join(PDF_CACHE_DIR, fname)
    if not os.path.exists(path):
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    return path

@st.cache_resource
def build_vectordb(items):
    if not items:
        return None
    docs = []
    for it in items:
        pdf_path = download_pdf(it["pdf_url"])
        for doc in PyPDFLoader(pdf_path).load():
            doc.metadata.update(it)
            docs.append(doc)
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks   = splitter.split_documents(docs)
    if not chunks:
        return None
    emb      = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, emb)

@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL),
        description="Answer Lok Sabha questions as the selected ministry.",
        instructions=[
            "Use PDF context to ground answers.",
            "Be formal, solution-oriented, positive, and focus on public welfare."
        ],
        show_tool_calls=False,
        markdown=False
    )

items = fetch_all_items()
if not items:
    st.error("No Q&A items found; site structure may have changed.")
    st.stop()

vectordb = build_vectordb(items)
if vectordb is None:
    st.error("Index build failed; no text extracted.")
    st.stop()

agent = init_agent()
ministries = sorted({i["ministry"] for i in items})
selected = st.selectbox("Select Ministry", ["All"] + ministries)

question = st.text_area("Enter your parliamentary question:")
if st.button("Get Answer"):
    if not question.strip():
        st.error("cannot answer this query")
        st.stop()
    docs = vectordb.similarity_search(question, k=10)
    if selected != "All":
        docs = [d for d in docs if d.metadata["ministry"] == selected]
    docs = docs[:4]
    if not docs:
        st.error("cannot answer this query")
        st.stop()
    context = "\n\n".join(
        f"[{d.metadata['session']} | {d.metadata['answer_date']}] {d.page_content.strip()}"
        for d in docs
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer as the Ministry of {selected}: Provide a formal, solution-oriented response "
        f"focusing on public interest. Include session, date, and source PDF link.\nQuestion: {question}"
    )
    response = agent.run(prompt)
    answer = getattr(response, "content", str(response))
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Sources")
    for d in docs:
        md = d.metadata
        st.markdown(f"- Session: {md['session']} | Date: {md['answer_date']}")
        st.markdown(f"  [PDF Source]({md['pdf_url']})")
