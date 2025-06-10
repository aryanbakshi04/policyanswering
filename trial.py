try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import json
import streamlit as st
import requests
from urllib.parse import urlencode

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration ---
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
def fetch_all_questions(lokNo=18, sessionNo=4, max_pages=625, page_size=10, locale="en"):
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

# --- Build FAISS vector store from filtered records ---
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

# --- Initialize Gemini Agent ---
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

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="Parliamentary Ministry Q&A Assistant",
        page_icon="🏛️",
        layout="wide"
    )

    st.title("🏛️ Parliamentary Ministry Q&A Assistant")

    if 'previous_questions' not in st.session_state:
        st.session_state.previous_questions = []

    # Load and cache all records once
    all_records = fetch_all_questions()
    ministries = sorted({rec['ministry'] for rec in all_records if rec['ministry']})
    
    if not ministries:
        st.error("No ministries found. Please try again later.")
        st.stop()

    # Sidebar for ministry selection
    with st.sidebar:
        st.header("Ministry Selection")
        selected_ministry = st.selectbox(
            "Choose Ministry",
            ministries,
            help="Select the ministry you want to query"
        )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_area(
            "Your Question to the Ministry:",
            height=100,
            help="Enter your question related to ministry affairs and public policy"
        )

        if st.button("🔍 Get Ministry Response", use_container_width=True):
            if not question.strip():
                st.error("Please enter a question.")
            elif not is_valid_question(question):
                st.error("Please ask a question related to ministry affairs and public policy.")
            else:
                if question not in st.session_state.previous_questions:
                    st.session_state.previous_questions.append(question)

                filtered = [r for r in all_records if r['ministry'] == selected_ministry]
                if not filtered:
                    st.error("No records found for selected ministry.")
                    st.stop()

                vectordb = build_vectorstore(filtered)
                if vectordb is None:
                    st.error("No searchable documents available for this ministry.")
                    st.stop()

                agent = init_agent()

                with st.spinner("Analyzing parliamentary records..."):
                    try:
                        docs = vectordb.similarity_search(question, k=5)
                        if not docs:
                            st.error("No relevant information found to answer this query.")
                        else:
                            context = "\n\n".join(d.page_content for d in docs)
                            prompt = construct_prompt(question, context, selected_ministry)
                            
                            response = agent.run(prompt)
                            answer = response.content if hasattr(response, 'content') else str(response)
                            
                            st.subheader("🏛️ Official Ministry Response")
                            st.markdown(answer)
                            
                            with st.expander("📋 Source Details", expanded=True):
                                sessions = {}
                                for doc in docs:
                                    session = doc.metadata.get('session', 'N/A')
                                    if session not in sessions:
                                        sessions[session] = {
                                            'dates': set(),
                                            'urls': set()
                                        }
                                    sessions[session]['dates'].add(doc.metadata.get('date', 'N/A'))
                                    sessions[session]['urls'].add(doc.metadata.get('source_url', ''))
                                
                                for session, details in sessions.items():
                                    st.markdown(f"**Parliament Session:** {session}")
                                    st.markdown(f"**Dates Referenced:** {', '.join(sorted(details['dates']))}")
                                    st.markdown("**Source Documents:**")
                                    for url in details['urls']:
                                        if url:
                                            st.markdown(f"- [📄 View Parliamentary Record]({url})")
                    
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
