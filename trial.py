import pysqlite3 as _sqlite3  
import sys
sys.modules['sqlite3'] = _sqlite3
import sqlite3
import time
from datetime import datetime, timezone, timedelta

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
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict
from urllib.parse import urlencode

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

PDF_CACHE_DIR = "pdf_cache_sansad"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
API_URL = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"
CHROMA_COLLECTION = "parliamentary_qa"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

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

def fetch_all_questions(lokNo=18, sessionNo=4, max_pages=625, page_size=10, locale="en"):
    all_questions = []
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Fetching parliamentary records..."):
        for page in range(1, max_pages + 1):
            progress = page / max_pages
            progress_bar.progress(progress)
            status_text.text(f"Fetching page {page} of {max_pages}...")

            params = {
                "loksabhaNo": lokNo,
                "sessionNumber": sessionNo,
                "pageNo": page,
                "locale": locale,
                "pageSize": page_size
            }

            try:
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
                        resp.raise_for_status()
                        break
                    except requests.exceptions.RequestException:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.warning(f"Skipping page {page} due to connection error")
                            continue
                        time.sleep(1)

                data = resp.json()
                
                if not data:
                    status_text.text("Completed fetching all available records.")
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
                    status_text.text("No more questions found.")
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

                if len(all_questions) % 100 == 0:
                    status_text.text(f"Processed {len(all_questions)} questions so far...")

            except Exception as e:
                st.warning(f"Error on page {page}: {str(e)}")
                continue

        status_text.text(f"Completed! Total questions fetched: {len(all_questions)}")
        progress_bar.progress(1.0)

    return all_questions

def setup_chromadb():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except:
        collection = client.create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_function
        )
    return client, collection

def data_storage_chromadb(collection, records):
    BATCH_SIZE = 500
    
    try:
        collection.delete(where={'ministry': {'$exists': True}})
    except Exception:
        pass

    total_batches = len(records) // BATCH_SIZE + 1
    progress_bar = st.progress(0)
    status_text = st.empty()

    for batch in range(0, len(records), BATCH_SIZE):
        progress = (batch // BATCH_SIZE + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(f"Storing batch {batch // BATCH_SIZE + 1} of {total_batches}...")

        batch_records = records[batch:batch + BATCH_SIZE]
        documents = []
        metadatas = []
        ids = []

        for rec in batch_records:
            doc_text = f"""
            Subject: {rec['subject']}
            Question: {rec['question_text']}
            Ministry: {rec['ministry']}
            Type: {rec['type']}
            Member: {rec['member']}
            Date: {rec['date']}
            """
            doc_id = f"{rec['question_no']}_{rec['session']}_{rec['loksabha']}"
            metadata = {
                "ministry": rec['ministry'],
                "date": rec['date'],
                "session": rec['session'],
                "pdf_url": rec['pdf_url'],
                "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            st.warning(f"Error storing batch {batch // BATCH_SIZE + 1}: {str(e)}")
            continue

    status_text.text("Data storage completed!")
    progress_bar.progress(1.0)

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

def main():
    st.set_page_config(
        page_title="Parliamentary Ministry Q&A Assistant",
        page_icon="🏛️",
        layout="wide"
    )
    
    st.markdown("**System Information**")
    st.markdown(f"- **Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted):** 2025-06-10 13:17:59")
    st.markdown(f"- **Current User's Login:** aryanbakshi04")
    st.markdown(f"- **Data Storage:** ChromaDB")

    st.title("🏛️ Parliamentary Ministry Q&A Assistant")

    if 'previous_questions' not in st.session_state:
        st.session_state.previous_questions = []
    
    if 'data_loading' not in st.session_state:
        st.session_state.data_loading = False

    try:
        client, collection = setup_chromadb()
    except Exception as e:
        st.error("Error initializing database. Please try again.")
        st.stop()

    if not st.session_state.data_loading:
        st.session_state.data_loading = True
        all_records = fetch_all_questions()
        with st.spinner("Updating data in ChromaDB..."):
            data_storage_chromadb(collection, all_records)
        st.session_state.data_loading = False

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
        
        if st.button("Refresh Data"):
            with st.spinner("Fetching latest data..."):
                start_time = time.time()
                new_records = fetch_all_questions()
                
                st.info(f"""
                Data Statistics:
                - Total Questions: {len(new_records)}
                - Unique Ministries: {len(set(r['ministry'] for r in new_records))}
                - Time Taken: {time.time() - start_time:.2f} seconds
                """)
                
                data_storage_chromadb(collection, new_records)
                st.success("Data refreshed successfully!")
                st.experimental_rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_area(
            "Your Question to the Ministry:",
            height=100,
            help="Enter your question related to ministry affairs"
        )

        if st.button("Get Ministry Response", use_container_width=True):
            if not is_valid_question(question):
                st.error("Please provide a complete question.")
                return
            
            if question not in st.session_state.previous_questions:
                st.session_state.previous_questions.append(question)
            
            try:
                results = collection.query(
                    query_texts=[question],
                    n_results=5,
                    where={"ministry": selected_ministry}
                )
                
                if not results['documents'][0]:
                    st.error("No relevant information found.")
                    return
                
                context = "\n\n".join(results['documents'][0])
                agent = init_agent()
                
                with st.spinner("Generating response..."):
                    prompt = construct_prompt(question, context, selected_ministry)
                    response = agent.run(prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                    st.subheader("Official Ministry Response")
                    st.markdown(answer)
                    
                    with st.expander("Source Details", expanded=True):
                        for metadata in results['metadatas'][0]:
                            st.markdown(f"**Parliament Session:** {metadata['session']}")
                            st.markdown(f"**Date:** {metadata['date']}")
                            if metadata.get('pdf_url'):
                                st.markdown(f"[View Parliamentary Record]({metadata['pdf_url']})")
            
            except Exception as e:
                st.error("Error generating response. Please try again.")

    with col2:
        if st.session_state.previous_questions:
            st.subheader("Recent Questions")
            for prev_q in st.session_state.previous_questions[-5:]:
                st.markdown(f"- {prev_q}")

if __name__ == "__main__":
    main()
