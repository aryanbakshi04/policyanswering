import os
import json
import streamlit as st
import requests
from datetime import datetime, timezone, timedelta
import time
import numpy as np
from typing import List, Dict
from urllib.parse import urlencode

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

# Constants
PDF_CACHE_DIR = "pdf_cache_sansad"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
API_URL = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"
FAISS_INDEX_PATH = "./faiss_index"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

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

def create_faiss_index(records):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    texts = []
    metadatas = []
    
    for record in records:
        text = f"""
        Subject: {record['subject']}
        Question: {record['question_text']}
        Ministry: {record['ministry']}
        Type: {record['type']}
        Member: {record['member']}
        Date: {record['date']}
        """
        texts.append(text)
        metadatas.append({
            "ministry": record['ministry'],
            "date": record['date'],
            "session": record['session'],
            "pdf_url": record['pdf_url']
        })
    
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    db.save_local(FAISS_INDEX_PATH)
    return db

def load_faiss_index():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings)

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
    
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown("**System Information**")
    st.markdown(f"- **Current Date and Time (UTC):** {current_time}")
    st.markdown(f"- **Current User's Login:** aryanbakshi04")
    st.markdown(f"- **Data Storage:** FAISS")

    st.title("🏛️ Parliamentary Ministry Q&A Assistant")

    if 'previous_questions' not in st.session_state:
        st.session_state.previous_questions = []

    if 'data_loading' not in st.session_state:
        st.session_state.data_loading = False

    if not st.session_state.data_loading:
        st.session_state.data_loading = True
        all_records = fetch_all_questions()
        with st.spinner("Creating FAISS index..."):
            db = create_faiss_index(all_records)
        st.session_state.data_loading = False
    else:
        try:
            db = load_faiss_index()
        except Exception as e:
            st.error("Error loading index. Please refresh the page.")
            st.stop()

    ministries = sorted(set(doc.metadata['ministry'] for doc in db.docstore._dict.values()))
    
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
        
        if st.button("🔄 Refresh Data"):
            with st.spinner("Fetching latest data..."):
                start_time = time.time()
                new_records = fetch_all_questions()
                
                st.info(f"""
                Data Statistics:
                - Total Questions: {len(new_records)}
                - Unique Ministries: {len(set(r['ministry'] for r in new_records))}
                - Time Taken: {time.time() - start_time:.2f} seconds
                """)
                
                db = create_faiss_index(new_records)
                st.success("Data refreshed successfully!")
                st.experimental_rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_area(
            "Your Question to the Ministry:",
            height=100,
            help="Enter your question related to ministry affairs"
        )

        if st.button("🔍 Get Ministry Response", use_container_width=True):
            if not is_valid_question(question):
                st.error("Please provide a complete question.")
                return
            
            if question not in st.session_state.previous_questions:
                st.session_state.previous_questions.append(question)
            
            try:
                results = db.similarity_search_with_score(
                    question,
                    k=5,
                    filter={'ministry': selected_ministry}
                )
                
                if not results:
                    st.error("No relevant information found.")
                    return
                
                context = "\n\n".join([doc.page_content for doc, score in results])
                agent = init_agent()
                
                with st.spinner("Generating response..."):
                    prompt = construct_prompt(question, context, selected_ministry)
                    response = agent.run(prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                    st.subheader("🏛️ Official Ministry Response")
                    st.markdown(answer)
                    
                    with st.expander("📋 Source Details", expanded=True):
                        for doc, score in results:
                            st.markdown(f"**Parliament Session:** {doc.metadata['session']}")
                            st.markdown(f"**Date:** {doc.metadata['date']}")
                            if doc.metadata.get('pdf_url'):
                                st.markdown(f"[📄 View Parliamentary Record]({doc.metadata['pdf_url']})")
            
            except Exception as e:
                st.error("Error generating response. Please try again.")

    with col2:
        if st.session_state.previous_questions:
            st.subheader("Recent Questions")
            for prev_q in st.session_state.previous_questions[-5:]:
                st.markdown(f"- {prev_q}")

if __name__ == "__main__":
    main()
