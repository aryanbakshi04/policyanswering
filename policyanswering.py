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

@st.cache_data(show_spinner="Fetching parliamentary questions...")
def fetch_all_items():
    items = []
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Fetch just the first page to start
    try:
        listing = f"{BASE_URL}"
        r = session.get(listing, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Find all rows in the main table
        rows = soup.select("table.table-striped tbody tr")
        
        for row in rows:
            try:
                # Get the last column which contains the PDF link
                last_column = row.select("td")[-1]
                pdf_tag = last_column.select_one("a[href$='.pdf']")
                
                if pdf_tag:
                    pdf_url = urljoin(BASE_URL, pdf_tag["href"])
                    
                    # Get subject from the third column
                    subject = row.select("td")[2].text.strip()
                    
                    items.append({
                        "pdf_url": pdf_url,
                        "subject": subject
                    })
            except Exception as e:
                st.warning(f"Skipping row due to error: {str(e)}")
                continue
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
    
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
                # Add subject to metadata
                doc.metadata["subject"] = it["subject"]
                doc.metadata["pdf_url"] = it["pdf_url"]
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
        description="Answer parliamentary questions based on official documents.",
        instructions=[
            "Use PDF context to ground answers.",
            "Be formal, solution-oriented, and focus on public welfare.",
            "Include the subject and source PDF link in your response"
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

# Create a list of subjects for filtering
subjects = sorted({i["subject"] for i in items})
selected_subject = st.selectbox("Filter by Subject", ["All"] + subjects, index=0)

question = st.text_area("Enter your parliamentary question:", height=150)
if st.button("Generate Answer", type="primary"):
    if not question.strip():
        st.error("Please enter a valid question")
        st.stop()
        
    with st.spinner("Searching knowledge base..."):
        try:
            docs = vectordb.similarity_search(question, k=10)
            
            # Filter by subject if selected
            if selected_subject != "All":
                docs = [d for d in docs if d.metadata["subject"] == selected_subject]
            
            if not docs:
                st.error("No relevant documents found for this question")
                st.stop()
                
            # Limit to top 4 documents
            docs = docs[:4]
            
            context = "\n\n".join(
                f"[Subject: {d.metadata['subject']}]\n{d.page_content.strip()}"
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
                f"Provide a formal, solution-oriented response based on the context above."
            )
            response = agent.run(prompt)
            answer = getattr(response, "content", str(response))
            
            st.subheader("Response")
            st.write(answer)
            
            st.subheader("Reference Sources")
            for d in docs:
                md = d.metadata
                st.markdown(f"**{md['subject']}**")
                st.markdown(f"- [Source PDF]({md['pdf_url']})")
                st.divider()
                
        except Exception as e:
            st.error(f"Answer generation failed: {str(e)}")
