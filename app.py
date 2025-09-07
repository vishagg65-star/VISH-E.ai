import asyncio
import sys
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

# Windows-specific fix for gRPC async loop
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv("E:/Langchain_LangGraph/RAG_Use/.env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY is missing in your .env file!")
    st.stop()

# ----------------------------
# Initialize LLM + Embeddings
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.5
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key,
    transport="rest"
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="💬 VISH-E.ai", layout="wide")

def set_background_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(rgba(255,255,255,0.95), rgba(255,255,255,0.95)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        .stChatMessage {{
            background: rgba(255, 255, 255, 0.7) !important;
            border-radius: 15px;
            padding: 12px;
            font-family: 'Segoe UI', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example background
set_background_image("vishai.png")

st.title("💬 VISH-E.ai")
st.write("Hi! I'm **VISH-E.ai**, your friendly AI assistant. Ask me anything, or upload PDFs to make my answers even smarter!")

# ----------------------------
# Session state setup
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ----------------------------
# Display Chat History
# ----------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"🙂 **You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(f"🤖 **VISH-E.ai:** {msg.content}")

# ----------------------------
# ----------------------------
# Chat Input & Streaming Response with fallback
# ----------------------------
user_query = st.chat_input("💡 Type your question here...")

if user_query:
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(f"🙂 **You:** {user_query}")

    # Placeholder for AI typing
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        source_info = ""  # Will indicate where info came from

        try:
            if st.session_state.qa_chain:
                # Ask RAG chain first (based on PDFs)
                result = st.session_state.qa_chain({
                    "question": user_query,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]

                # Check if answer contains info from PDF or is generic
                if "I could not find" in answer or "Sorry" in answer.lower():
                    source_info = "⚠️ Note: This information is outside your uploaded PDFs."
                    # Fallback to plain LLM
                    answer = llm.invoke(user_query).content
                else:
                    source_info = "📄 Answer based on your uploaded PDFs."
            else:
                # No PDFs uploaded, use plain LLM
                answer = llm.invoke(user_query).content
                source_info = "⚠️ Note: This information is from my general knowledge."
        except Exception as e:
            # Safety fallback
            answer = llm.invoke(user_query).content
            source_info = "⚠️ Note: This information is from my general knowledge."

        # Stream the answer word by word
        for word in answer.split():
            full_response += word + " "
            placeholder.markdown(f"🤖 **VISH-E.ai:** {full_response}")
            asyncio.run(asyncio.sleep(0.04 + 0.02 * (len(word)/5)))  # human-like typing

        # Append source info at the end
        full_response += f"\n\n{source_info}"
        placeholder.markdown(f"🤖 **VISH-E.ai:** {full_response}")

    # Save AI response
    st.session_state.chat_history.append(AIMessage(content=full_response))


       

    # Save AI response
    st.session_state.chat_history.append(AIMessage(content=full_response))

# ----------------------------
# PDF Uploader
# ----------------------------
st.divider()
st.subheader("📂 Upload PDFs and explore ,find anything within seconds!")
uploaded_files = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        # Save locally
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        all_docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(all_docs)

    # Create vector DB
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Create RAG chain
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    st.success("✅ Done! I'm now smarter and ready to chat using your PDFs.")

