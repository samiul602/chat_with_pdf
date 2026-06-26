# 📄 Chat with PDF — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about its content. Powered by Groq LLaMA 3.1 and FAISS vector search.

🔗 **Live Demo:** https://chat-with-anypdf.streamlit.app/

---

## 🚀 Features

- 📤 Upload any PDF file
- 🔍 Automatic text extraction and chunking
- 🧠 Semantic search using sentence embeddings
- 💬 Conversational Q&A powered by Groq LLaMA 3.1
- 🗑️ Clear conversation history anytime
- 📊 PDF statistics in sidebar

---

## ⚙️ How it Works

```
Upload PDF → Extract Text → Split into Chunks → Create Embeddings
→ Store in FAISS → User asks question → Search FAISS
→ Retrieve top chunks → Groq LLM generates answer
```

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| PDF Reading | PyMuPDF |
| Chunking | LangChain Text Splitters |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| LLM | Groq (LLaMA 3.1 8b Instant) |

---

## 🏃 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/samiul602/chat_with_pdf.git
cd chat_with_pdf
```

**2. Create virtual environment**
```bash
py -3.11 -m venv cwp_env
cwp_env\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up API key**

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
chat_with_pdf/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── .env                    # API key (not pushed to GitHub)
├── .gitignore
└── README.md
```

---

## 👨‍💻 Developer

**Samiul** — CSE Student, North South University (NSU)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).