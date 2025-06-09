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

BASE_URL = "https://sansad.in/ls/questions/questions-and-answers"
CACHE_DIR = "pdf_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash-exp"

os.makedirs(CACHE_DIR, exist_ok=True)

def detect_pages():
    r = requests.get(BASE_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pager = soup.select_one("ul.pagination")
    if not pager:
        return 1
    nums = [int(a.text) for a in pager.select("a") if a.text.isdigit()]
    return max(nums) if nums else 1

@st.cache_data
def fetch_items():
    items, seen = [], set()
    total = detect_pages()
    for page in range(1, total+1):
        url = f"{BASE_URL}?page={page}"
        r = requests.get(url); r.raise_for_status()
        rows = BeautifulSoup(r.text, "html.parser")\
               .select("table.table-striped tbody tr")
        if not rows:
            break
        for row in rows:
            link = row.select_one("td:nth-of-type(5) a")
            if not link: continue
            detail = urljoin(BASE_URL, link["href"])
            dr = requests.get(detail); dr.raise_for_status()
            ds = BeautifulSoup(dr.text, "html.parser")

            pdf = ds.select_one("a[href$='.pdf']")
            if not pdf: continue
            pdf_url = urljoin(detail, pdf["href"])
            if pdf_url in seen: continue
            seen.add(pdf_url)

            mins = ds.find("th", text="Ministry").find_next_sibling("td").text.strip()
            sess = ds.find("th", text="Session").find_next_sibling("td").text.strip()
            date= ds.find("th", text="Answer Date").find_next_sibling("td").text.strip()

            items.append({
                "pdf_url": pdf_url,
                "ministry": mins,
                "session": sess,
                "answer_date": date
            })
    return items

@st.cache_data
def download_pdf(url):
    fn = os.path.basename(url)
    path = os.path.join(CACHE_DIR, fn)
    if not os.path.exists(path):
        r = requests.get(url); r.raise_for_status()
        open(path, "wb").write(r.content)
    return path

@st.cache_resource
def build_db(itms):
    if not itms: return None
    docs=[]
    for x in itms:
        p = download_pdf(x["pdf_url"])
        for d in PyPDFLoader(p).load():
            d.metadata.update(x); docs.append(d)
    if not docs: return None
    chunks = RecursiveCharacterTextSplitter(1000,100).split_documents(docs)
    if not chunks: return None
    emb = SentenceTransformerEmbeddings(EMBED_MODEL)
    return FAISS.from_documents(chunks, emb)

@st.cache_resource
def make_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL),
        description="Answer as Ministry, formal & solution-oriented.",
        instructions=["Use PDF context, focus on public welfare."],
        show_tool_calls=False
    )

items = fetch_items()
if not items:
    st.error("No Q&A items found; site structure may have changed.")
    st.stop()

db = build_db(items)
if not db:
    st.error("Index build failed; no text extracted.")
    st.stop()

agent = make_agent()
mins = sorted({i["ministry"] for i in items})
sel = st.selectbox("Select Ministry", ["All"]+mins)

q = st.text_area("Your parliamentary question:")
if st.button("Get Answer"):
    if not q.strip():
        st.error("cannot answer this query"); st.stop()
    docs = db.similarity_search(q, k=10)
    if sel!="All":
        docs = [d for d in docs if d.metadata["ministry"]==sel]
    docs=docs[:4]
    if not docs:
        st.error("cannot answer this query"); st.stop()
    ctx = "\n\n".join(f"[{d.metadata['session']} | {d.metadata['answer_date']}] {d.page_content.strip()}" for d in docs)
    prompt = f"Context:\n{ctx}\n\nAnswer as the Ministry of {sel}: formal, solution-oriented, include session, date & PDF link.\nQuestion: {q}"
    res = agent.run(prompt)
    ans = getattr(res,"content",str(res))
    st.subheader("Answer"); st.write(ans)
    st.subheader("Sources")
    for d in docs:
        md=d.metadata
        st.markdown(f"- Session: {md['session']} | Date: {md['answer_date']}")
        st.markdown(f"  [PDF]({md['pdf_url']})")
