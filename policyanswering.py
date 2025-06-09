try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os, streamlit as st, requests, time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

BASE_URL = "https://sansad.in/ls/questions/questions-and-answers"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
}
PDF_CACHE_DIR = "pdf_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

def detect_total_pages():
    try:
        resp = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        pager = soup.select_one("ul.pagination")
        if not pager:
            return 1
            
        last_page_link = pager.select("li.page-item a")[-1]
        if 'href' in last_page_link.attrs:
            last_page_url = last_page_link['href']
            page_param = last_page_url.split('page=')[-1]
            return int(page_param)
        return 1
    except Exception as e:
        st.warning(f"Couldn't detect total pages: {str(e)}. Using default 10 pages")
        return 10

@st.cache_data(show_spinner="Fetching parliamentary questions...")
def fetch_all_items():
    total = detect_total_pages()
    items, seen = [], set()
    session = requests.Session()
    session.headers.update(HEADERS)
    
    for pg in range(1, total + 1):
        try:
            listing = f"{BASE_URL}?page={pg}"
            r = session.get(listing, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            
            table = soup.select_one("table.table-striped")
            if not table:
                st.warning(f"No table found on page {pg}. Site structure may have changed.")
                continue
                
            rows = table.select("tbody tr")
            if not rows:
                st.info(f"No rows found on page {pg}. Stopping collection.")
                break
                
            for row in rows:
                try:
                    link_cell = row.select_one("td:nth-child(1)") or row.select_one("td:first-child")
                    if not link_cell:
                        continue
                    link = link_cell.find("a")
                    if not link or 'href' not in link.attrs:
                        continue
                        
                    detail_url = urljoin(BASE_URL, link["href"])
                    dr = session.get(detail_url, timeout=10)
                    dr.raise_for_status()
                    ds = BeautifulSoup(dr.text, "html.parser")
                    
                    pdf_tag = ds.select_one("a[href$='.pdf']")
                    if not pdf_tag:
                        continue
                        
                    pdf_url = urljoin(detail_url, pdf_tag["href"])
                    if pdf_url in seen:
                        continue
                    seen.add(pdf_url)
                    
                    metadata = {
                        "pdf_url": pdf_url,
                        "ministry": "Unknown",
                        "session": "Unknown",
                        "answer_date": "Unknown"
                    }
                    
                    info_table = ds.select_one("table.table")
                    if info_table:
                        for tr in info_table.select("tr"):
                            th = tr.select_one("th")
                            td = tr.select_one("td")
                            if th and td:
                                key = th.text.strip().lower()
                                value = td.text.strip()
                                if "ministry" in key:
                                    metadata["ministry"] = value
                                elif "session" in key:
                                    metadata["session"] = value
                                elif "answer date" in key:
                                    metadata["answer_date"] = value
                    
                    items.append(metadata)
                    time.sleep(0.5)
                except Exception as e:
                    st.warning(f"Skipping row due to error: {str(e)}")
        except Exception as e:
            st.warning(f"Skipping page {pg} due to error: {str(e)}")
            
    return items

@st.cache_data
def download_pdf(url):
    fname = os.path.basename(url.split("?")[0])
    path = os.path.join(PDF_CACHE_DIR, fname)
    if not os.path.exists(path):
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    return path

def safe_pdf_loader(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        if docs and docs[0].page_content.strip():
            return docs
        loader = UnstructuredPDFLoader(pdf_path, strategy="ocr_only")
        return loader.load()
    except Exception as e:
        st.warning(f"PDF loader failed for {os.path.basename(pdf_path)}: {str(e)}")
        return []

@st.cache_resource(show_spinner="Building knowledge base...")
def build_vectordb(items):
    if not items:
        return None
    docs = []
    for it in items:
        try:
            pdf_path = download_pdf(it["pdf_url"])
            doc_chunks = safe_pdf_loader(pdf_path)
            for doc in doc_chunks:
                doc.metadata.update(it)
                docs.append(doc)
        except Exception as e:
            st.warning(f"Skipping PDF {it['pdf_url']}: {str(e)}")
    
    if not docs:
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None
        
    emb = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, emb)

@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL),
        description="Answer Lok Sabha questions as the selected ministry.",
        instructions=[
            "Use PDF context to ground answers.",
            "Be formal, solution-oriented, positive, and focus on public welfare.",
            "Include session, date, and source PDF link in your response"
        ],
        show_tool_calls=False,
        markdown=False
    )

st.title("Parliamentary Q&A System")
st.caption("Answers based on historical Lok Sabha questions and answers")

with st.spinner("Initializing system..."):
    items = fetch_all_items()
    if not items:
        st.error("No Q&A items found. Please try again later or check the website structure.")
        st.stop()

    vectordb = build_vectordb(items)
    if vectordb is None:
        st.error("Failed to build knowledge base. No valid documents processed.")
        st.stop()

    agent = init_agent()

ministries = sorted({i["ministry"] for i in items if i["ministry"] != "Unknown"})
selected = st.selectbox("Select Ministry", ["All"] + ministries, index=0)

question = st.text_area("Enter your parliamentary question:", height=150)
if st.button("Generate Answer", type="primary"):
    if not question.strip():
        st.error("Please enter a valid question")
        st.stop()
        
    with st.spinner("Searching knowledge base..."):
        try:
            docs = vectordb.similarity_search(question, k=10)
            if selected != "All":
                docs = [d for d in docs if d.metadata["ministry"] == selected]
            docs = docs[:4]
            
            if not docs:
                st.error("No relevant documents found for this question")
                st.stop()
                
            context = "\n\n".join(
                f"[{d.metadata.get('session', 'Unknown')} | {d.metadata.get('answer_date', 'Unknown')}] {d.page_content.strip()}"
                for d in docs
            )
        except Exception as e:
            st.error(f"Knowledge base search failed: {str(e)}")
            st.stop()
            
    with st.spinner("Generating answer..."):
        try:
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer as the Ministry of {selected}:"
            )
            response = agent.run(prompt)
            answer = getattr(response, "content", str(response))
            
            st.subheader("Ministry's Response")
            st.write(answer)
            
            st.subheader("Reference Sources")
            for d in docs:
                md = d.metadata
                st.markdown(f"**{md.get('ministry', 'Unknown Ministry')}**")
                st.markdown(f"- Session: {md.get('session', 'Unknown')}")
                st.markdown(f"- Date Answered: {md.get('answer_date', 'Unknown')}")
                st.markdown(f"- [Source PDF]({md['pdf_url']})")
                st.divider()
                
        except Exception as e:
            st.error(f"Answer generation failed: {str(e)}")
