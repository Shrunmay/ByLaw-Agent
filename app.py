import streamlit as st
import sys
import os

# Add the 'modules' directory to the Python path
# This allows us to import the files directly
sys.path.append(os.path.join(os.getcwd(), "modules"))

from retriever import load_brain, get_city_context
from llm_interface import process_query

# --- PAGE CONFIG ---
st.set_page_config(page_title="Toronto City AI", page_icon="🏙️", layout="centered")

# --- CUSTOM CSS FOR POLISH ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'vectorstore' not in st.session_state:
    with st.spinner("Initializing 30k City Records..."):
        # Ensure it looks for the local chroma_db folder
        db_path = os.path.join(os.getcwd(), "chroma_db")
        st.session_state.vectorstore = load_brain(db_path)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR (Project Metadata for TA) ---
with st.sidebar:
    st.title("📊 Agent Status")
    st.info("Knowledge Base: 30,000 Records")
    st.success("Model: Qwen-30B Reasoning")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- CHAT INTERFACE ---
st.title("🏙️ Toronto City Bylaw Agent")
st.markdown("How can I help you with **Hazards**, **Permits**, or **Waste** today?")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("Type your question here..."):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. RAG Retrieval (Retriever Module)
    with st.spinner("Searching city archives..."):
        context, objective = get_city_context(user_input, st.session_state.vectorstore)
    
    # 3. LLM Reasoning (Interface Module)
    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            # Pass the query, retrieved text, the department (objective), and history
            response = process_query(
                user_input, 
                context, 
                objective, 
                st.session_state.messages[:-1]
            )
            st.markdown(response)
            
            # Source Attribution (Requirement #2)
            if objective:
                st.caption(f"Source: Toronto Open Data - {objective}")

    # 4. Save Assistant Response to Memory
    st.session_state.messages.append({"role": "assistant", "content": response})
