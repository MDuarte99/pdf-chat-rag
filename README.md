# 📄 PDF Chat with RAG (Retrieval-Augmented Generation)

An end-to-end AI application that enables natural language interaction with PDF documents using a **Retrieval-Augmented Generation (RAG)** pipeline powered by local LLMs.

This project showcases how to design and implement a **production-oriented RAG system**, combining efficient document retrieval with context-aware text generation — fully running locally, without external APIs.

---

## 🚀 Features

- 📄 **PDF Ingestion & Processing** — Extracts and preprocesses document text  
- ✂️ **Intelligent Chunking** — Splits text into semantically meaningful segments  
- 🧬 **Embedding Generation** — Converts text into dense vector representations  
- 🔍 **Semantic Retrieval** — Efficient similarity search using vector indexing  
- 🤖 **Context-Aware Generation** — LLM responses grounded in retrieved content  
- 💻 **Fully Local Execution** — No dependency on paid APIs or external services  

---

## 🧠 Architecture

```
PDF → Text Extraction → Chunking → Embeddings → Vector Store → Retrieval → LLM → Answer
```

This pipeline ensures that responses are **grounded, explainable, and contextually relevant**, reducing hallucinations typically seen in standalone LLM applications.

---

## 🛠️ Tech Stack

- **Python** — Core development  
- **LangChain** — RAG pipeline orchestration  
- **FAISS** — High-performance vector similarity search  
- **Ollama** — Local LLM runtime  
- **Llama 3** — Open-source large language model  

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/pdf-chat-rag.git
cd pdf-chat-rag
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the local LLM

Ensure Ollama is installed and running:

```bash
ollama run llama3
```

### 5. Add your PDF

Place your document at:

```
/pdf/sample.pdf
```

### 6. Run the application

```bash
python main.py
```

---

## 💬 Example Usage

**User Input:**
```
What is the document about?
```

**Model Output:**
```
This document provides an overview of...
```

---

## 📌 Core Concepts

### 🔹 Retrieval-Augmented Generation (RAG)
Combines information retrieval with language generation, allowing the model to answer questions using **external, domain-specific data**.

### 🔹 Embeddings
Transforms text into numerical vectors, enabling **semantic similarity search** instead of keyword matching.

### 🔹 Vector Database
Stores embeddings and supports **fast, scalable retrieval** of relevant context.

---

## 🚧 Future Improvements

- 🌐 API layer with FastAPI  
- 🖥️ Web-based user interface  
- 📚 Multi-document support  
- ⚡ Streaming responses  
- 📊 Evaluation & monitoring (e.g., RAG metrics, latency, accuracy)  

---

## 📸 Demo

*(Add a screenshot or GIF demonstrating the interaction flow)*

---

## 🤝 Contributing

Contributions are welcome. Feel free to open issues, suggest improvements, or submit pull requests.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Developed as part of an **AI Engineering learning path**, focusing on building **scalable, production-ready LLM applications** with real-world relevance.
