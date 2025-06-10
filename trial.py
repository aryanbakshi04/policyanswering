try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import json
import streamlit as st
import requests
import chromadb
from chromadb.config import settings
from chromadb.utils import embedding_functions
from typing import List,Dict
from datetime import datetime,timezone
from urllib.parse import urlencode

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini


PDF_CACHE_DIR = "pdf_cache_sansad"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
API_URL = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

def is_valid_question(question: str) -> bool:
    question = question.strip()
    words = question.split()
    
    return len(question) > 0 and len(words) >= 3

def construct_prompt(question, context, ministry):
    return f"""
Context from Parliamentary Records:
{context}

Instructions:
- Provide a formal response on behalf of the {ministry}
- Focus on public welfare and practical solutions
- Maintain a positive, constructive tone
- Include relevant policies and initiatives
- Only answer if the question is relevant to ministry affairs

Question: {question}

Response Format:
1. Formal Ministry Response
2. Related Initiatives (if any)
3. Future Plans/Recommendations (if applicable)
"""

@st.cache_data(ttl=24*3600)
def fetch_all_questions(lokNo=18, sessionNo=4, max_pages=2, page_size=10, locale="en"):
    all_questions = []
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    with st.spinner("Fetching parliamentary records..."):
        for page in range(1, max_pages):
            params = {
                "loksabhaNo": lokNo,
                "sessionNumber": sessionNo,
                "pageNo": page,
                "locale": locale,
                "pageSize": page_size
            }

            try:
                resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    break

                questions = []
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        qlist = item.get("listOfQuestions", [])
                        if isinstance(qlist, list):
                            valid_questions = [
                                q for q in qlist 
                                if isinstance(q, dict) and q.get("ministry")
                            ]
                            questions.extend(valid_questions)

                if not questions:
                    break

                for q in questions:
                    ministry = q.get("ministry")
                    if not ministry:
                        continue

                    processed_q = {
                        "question_no": q.get("quesNo"),
                        "subject": q.get("subjects"),
                        "loksabha": q.get("lokNo"),
                        "session": q.get("sessionNo"),
                        "member": (", ".join(q.get("member", []))
                                 if isinstance(q.get("member"), list)
                                 else q.get("member")),
                        "ministry": ministry,
                        "type": q.get("type"),
                        "pdf_url": q.get("questionsFilePath"),
                        "question_text": q.get("questionText"),
                        "date": q.get("date"),
                    }
                    all_questions.append(processed_q)

            except Exception as e:
                break

    return all_questions


@st.cache_resource
def build_vectorstore(records):
    docs = []
    with st.spinner("Processing ministry documents..."):
        for rec in records:
            if rec['pdf_url']:
                fname = os.path.join(PDF_CACHE_DIR, os.path.basename(rec['pdf_url']))
                if not os.path.exists(fname):
                    try:
                        r = requests.get(rec['pdf_url'], timeout=30)
                        r.raise_for_status()
                        with open(fname, 'wb') as f:
                            f.write(r.content)
                    except Exception:
                        continue
                        
                try:
                    loader = PyPDFLoader(fname)
                    loaded = loader.load()
                    for d in loaded:
                        d.metadata.update({
                            'session': rec['session'],
                            'date': rec['date'],
                            'ministry': rec['ministry'],
                            'source_url': rec['pdf_url']
                        })
                    docs.extend(loaded)
                except Exception:
                    continue
                    
        if not docs:
            return None
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.from_documents(chunks, embeddings)


@st.cache_resource
def init_agent():
    return Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Official Parliamentary Ministry Q&A Assistant providing formal, solution-oriented responses.",
        instructions=[
            "Focus solely on public interest and welfare when answering questions.",
            "Provide solution-oriented responses with a positive tone.",
            "Only answer questions relevant to ministry affairs and public policy.",
            "Return 'Cannot answer this query. Please ask a question related to ministry affairs and public policy.' for irrelevant questions.",
            "Include specific details from source documents to support answers.",
            "Maintain formal, parliamentary language throughout responses.",
            "Structure responses to clearly address the question's main points.",
            "Cite relevant dates and sessions when referencing similar questions or policies."
        ],
        show_tool_calls=False,
        markdown=True
    )

chroma_collection="parliamentary_Q&A"

def setup_chromadb():
    client=chromadb.HttpClient(
        host=os.getenv(
    )
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    try:
        chroma_collection=client.get_collection(chroma_collection)
    except:
        chroma_collection=client.create_collection(
            name=chroma_collection,
            embedding_function=embedding_function
        )
    return client,chroma_collection

def data_storage_chromadb(collection,records):
    for batch in range(0,len(records),100):
        batch_records=records[batch:batch+100]

        documents=[]
        metadatas=[]
        ids=[]

        for rec in batch_records:
            doc_text=f"""
            Subject:{rec['subject']}
            Ministry:{rec['ministry']}
            Type:{rec['type']}
            Member:{rec['member']}
            Date:{rec['date']}
            """
            doc_id=f"{rec['question_no']}_{rec['session']}_{rec['loksabha']}"
            metadata={
                "ministry": rec['ministry'],
                "date": rec['date'],
                "session": rec['session'],
                "pdf_url": rec['pdf_url']
            }
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids)
            
        except Exception as e:
            continue

def main():
    st.set_page_config(
        page_title="Parliamentary Ministry Q&A Assistant",
        page_icon="🏛️",
        layout="wide"
    )
        
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"**Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted):** {current_time}")
    st.markdown(f"**Current User's Login:** aryanbakshi04")

    st.title("🏛️ Parliamentary Ministry Q&A Assistant")

    if 'previous_questions' not in st.session_state:
        st.session_state.previous_questions = []
    
    client,chroma_collection=setup_chromadb()
    
    all_records = fetch_all_questions()
    if not st.session_state.get('data_stored'):
        with st.spinner("Storing data in ChromaDB..."):
            data_storage_chromadb(chroma_collection, all_records)
            st.session_state.data_stored = True

    ministries = sorted({rec['ministry'] for rec in all_records if rec['ministry']})
    
    if not ministries:
        st.error("No ministries found. Please try again later.")
        st.stop()

    
    with st.sidebar:
        st.header("Ministry Selection")
        selected_ministry = st.selectbox(
            "Choose Ministry",
            ministries,
            help="Select the ministry you want to query"
        )

    
    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_area(
            "Your Question to the Ministry:",
            height=100,
            help="Enter your question related to ministry affairs and public policy"
        )

        if st.button("Get Ministry Response", use_container_width=True):
            if not question.strip():
                st.error("Please enter a question.")
            elif not is_valid_question(question):
                st.error("Please ask a question related to ministry affairs and public policy.")
            else:
                if question not in st.session_state.previous_questions:
                    st.session_state.previous_questions.append(question)

                try:
                    results = chroma_collection.query(
                        query_texts=[question],
                        n_results=5,
                        where={"ministry": selected_ministry}
                    )
                    
                    if not results['documents'][0]:
                        st.error("No relevant information found.")
                        st.stop()
                    
                    context = "\n\n".join(results['documents'][0])
                    agent = init_agent()
                    
                    with st.spinner("Generating response..."):
                        prompt = construct_prompt(question, context, selected_ministry)
                        response = agent.run(prompt)
                        answer = response.content if hasattr(response, 'content') else str(response)
                        
                        st.subheader("🏛️ Official Ministry Response")
                        st.markdown(answer)
                        
                        with st.expander("📋 Source Details", expanded=True):
                            for metadata in results['metadatas'][0]:
                                st.markdown(f"**Parliament Session:** {metadata['session']}")
                                st.markdown(f"**Date:** {metadata['date']}")
                                if metadata.get('pdf_url'):
                                    st.markdown(f"[📄 View Parliamentary Record]({metadata['pdf_url']})")
                
                except Exception as e:
                    st.error("Error generating response. Please try again.")
                    st.stop()

    with col2:
        if st.session_state.previous_questions:
            st.subheader("Recent Questions")
            for prev_q in st.session_state.previous_questions[-5:]:
                st.markdown(f"- {prev_q}")

if __name__ == "__main__":
    main()
