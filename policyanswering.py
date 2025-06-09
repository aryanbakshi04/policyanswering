try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
PDF_CACHE_DIR = "pdf_cache_sansad"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

@st.cache_data(ttl=24*3600)
def fetch_ministries():
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
  
    select = soup.find('select', {'id': 'edit-field-department-tid'})
    options = select.find_all('option') if select else []
    return [opt.text.strip() for opt in options if opt.get('value')]

@st.cache_data(ttl=24*3600)
def fetch_qna_records(ministry):
    records = []
    import urllib.parse
    page = 0
    while True:
        params = {'field_department_tid': ministry, 'page': page}
        resp = requests.get(BASE_URL, params=params)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, 'html.parser')
        rows = soup.select('div.view-content .views-row')
        if not rows:
            break
        for row in rows:
            
            q = row.select_one('.views-field-field-question-content .field-content')
           
            s = row.select_one('.views-field-field-parallel-session-tid .field-content')
      
            d = row.select_one('.views-field-created .field-content')
          
            link = row.select_one('.views-field-field-version-file a[href$=".pdf"]')
            pdf_url = urllib.parse.urljoin(BASE_URL, link['href']) if link else None
            if q and s and d:
                records.append({
                    'question': q.get_text(strip=True),
                    'session': s.get_text(strip=True),
                    'date': d.get_text(strip=True),
                    'pdf_url': pdf_url
                })
        page += 1
    return records

@st.cache_resource
def build_vectorstore(records):
    docs = []
    for rec in records:
        if rec['pdf_url']:
            fname = os.path.join(PDF_CACHE_DIR, os.path.basename(rec['pdf_url']))
            if not os.path.exists(fname):
                r = requests.get(rec['pdf_url']); r.raise_for_status()
                with open(fname, 'wb') as f: f.write(r.content)
            loader = PyPDFLoader(fname)
            loaded = loader.load()
            for d in loaded:
                d.metadata.update({'session': rec['session'], 'date': rec['date'], 'source_url': rec['pdf_url']})
            docs.extend(loaded)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Answers parliamentary ministry questions based on retrieved context.",
        instructions=[
            "Use context from ministry Q&A PDFs.",
            "Provide formal, solution-oriented responses focused on public welfare."
        ],
        show_tool_calls=False,
        markdown=False
    )


st.title("Parliamentary Ministry Q&A Assistant")
ministry_list = fetch_ministries()
selected_ministry = st.sidebar.selectbox("Select Ministry", ministry_list)

if 'ministry' not in st.session_state or st.session_state['ministry'] != selected_ministry:
    with st.spinner(f"Indexing Q&A for {selected_ministry}..."):
        records = fetch_qna_records(selected_ministry)
        if not records:
            st.error("cannot answer this query")
            st.stop()
        vectordb = build_vectorstore(records)
        st.session_state.vectordb = vectordb
        st.session_state.agent = init_agent()
        st.session_state.ministry = selected_ministry

question = st.text_area("Your Parliamentary Question:")
if st.button("Get Ministry Response"):
    if not question:
        st.error("cannot answer this query")
    else:
        docs = st.session_state.vectordb.similarity_search(question, k=5)
        if not docs:
            st.error("cannot answer this query")
        else:
            context = "\n\n".join(d.page_content for d in docs)
            prompt = (
                f"Context:\n{context}\n\n"
                f"Provide answer as {selected_ministry} ministry. Include session and date. Question: {question}"
            )
            response = st.session_state.agent.run(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            st.subheader("Response")
            st.write(answer)
            st.subheader("Details")
            st.write(f"**Session:** {docs[0].metadata.get('session')}  ")
            st.write(f"**Date:** {docs[0].metadata.get('date')}  ")
            st.subheader("Source PDFs")
            for url in {d.metadata.get('source_url') for d in docs}:
                if url: st.markdown(f"- [Download PDF]({url})")
