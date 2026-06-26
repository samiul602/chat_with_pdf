import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ─── PDF Text Extraction ───────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ─── Text Chunking ─────────────────────────────────────────────────
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# ─── FAISS Vector Store ────────────────────────────────────────────
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def search_faiss(query, index, chunks, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# ─── Groq LLM Answer ───────────────────────────────────────────────
def get_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question based only on the provided context.
If the answer is not in the context, say "I couldn't find this information in the PDF."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pdf.png", width=80)
    st.title("📄 Chat with PDF")
    st.markdown("---")

    st.markdown("### 🤖 About this App")
    st.info(
        "This app lets you upload any PDF and ask questions about it. "
        "It uses AI to find relevant information and generate accurate answers."
    )

    st.markdown("### ⚙️ How it Works")
    st.markdown("""
1. 📤 **Upload** a PDF file
2. 🔍 **Indexing** — text is extracted and split into chunks
3. 🧠 **Embeddings** — chunks are converted to vectors using `all-MiniLM-L6-v2`
4. 💾 **FAISS** — vectors are stored in a vector database
5. ❓ **Ask** your question
6. 🔎 **Search** — top relevant chunks are retrieved
7. 💬 **Answer** — Groq LLM generates the final answer
    """)

    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
| Component | Tool |
|-----------|------|
| UI | Streamlit |
| PDF Reading | PyMuPDF |
| Chunking | LangChain |
| Embeddings | sentence-transformers |
| Vector DB | FAISS |
| LLM | Groq (LLaMA 3.1) |
    """)

    st.markdown("---")

    # PDF stats (shown after upload)
    if "pdf_stats" in st.session_state:
        st.markdown("### 📊 PDF Statistics")
        st.metric("Total Chunks", st.session_state.pdf_stats["chunks"])
        st.metric("Total Characters", st.session_state.pdf_stats["chars"])
        st.metric("File Name", st.session_state.pdf_stats["name"])

    st.markdown("---")
    st.markdown("### 🗑️ Reset Chat")
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & Groq")

# ─── Main UI ──────────────────────────────────────────────────────
st.title("📄 Chat with your PDF")
st.markdown("Upload a PDF file and ask any question about its content.")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("⏳ Reading and indexing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, _ = create_faiss_index(chunks)

        # Save stats to session state
        st.session_state.pdf_stats = {
            "chunks": len(chunks),
            "chars": len(text),
            "name": uploaded_file.name
        }

    st.success(f"✅ **{uploaded_file.name}** indexed! Chunks: **{len(chunks)}**")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if question := st.chat_input("💬 Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                context_chunks = search_faiss(question, index, chunks)
                answer = get_answer(question, context_chunks)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Please upload a PDF file to get started.")