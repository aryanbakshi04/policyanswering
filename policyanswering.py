try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os, streamlit as st, requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from agno.agent import Agent
from agno.models.google import Gemini

API_URL       = "https://sansad.in/api/question/getLSQPage"
PDF_CACHE_DIR = "pdf_cache"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL  = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

@st.cache_data
def fetch_all_items():
    items, seen = [], set()
    page = 1
    while True:
        resp = requests.get(f"{API_URL}?page={page}&lang=en&type=LSQ")
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        for row in data:
            pdf_url  = row.get("pdf", "").strip()
            ministry = row.get("ministryName", "").strip()
            session  = (row.get("lakshadSession") or row.get("session") or "").strip()
            date     = row.get("date", "").strip()
            if not pdf_url or pdf_url in seen:
                continue
            seen.add(pdf_url)
            items.append({
                "pdf_url":     pdf_url,
                "ministry":    ministry,
                "session":     session,
                "answer_date": date
            })
        page += 1
    return items

@st.cache_data
def download_pdf(url):
    fname = os.path.basename(url)
    path = os.path.join(PDF_CACHE_DIR, fname)
    if not os.path.exists(path):
        r = requests.get(url)
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
        path = download_pdf(it["pdf_url"])
        for doc in PyPDFLoader(path).load():
            doc.metadata.update(it)
            docs.append(doc)
    if not docs:
        return None
    chunks    = RecursiveCharacterTextSplitter(1000, 100).split_documents(docs)
    if not chunks:
        return None
    embedder  = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embedder)

@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL),
        description="Answer Lok Sabha questions as the selected ministry.",
        instructions=[
            "Use the PDF context for factual grounding.",
            "Be formal, positive, solution-oriented, and focus on public welfare."
        ],
        show_tool_calls=False,
        markdown=False
    )

items = fetch_all_items()
if not items:
    st.error("No Q&A items fetched; endpoint or parameters may have changed.")
    st.stop()

vectordb = build_vectordb(items)
if vectordb is None:
    st.error("Index build failed; no text extracted.")
    st.stop()

agent    = init_agent()
ministries = sorted({i["ministry"] for i in items})
selected   = st.selectbox("Select Ministry", ["All"] + ministries)

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
    answer   = getattr(response, "content", str(response))
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Sources")
    for d in docs:
        md = d.metadata
        st.markdown(f"- Session: {md['session']} | Date: {md['answer_date']}")
        st.markdown(f"  [PDF Source]({md['pdf_url']})")
