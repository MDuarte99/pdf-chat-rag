# Import responsible for loading and parsing PDF documents into structured text
from langchain_community.document_loaders import PyPDFLoader

# Responsible for splitting long documents into smaller chunks to fit LLM context limits
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + LLM interface for local inference via Ollama
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Vector database used for semantic search over embeddings
from langchain_community.vectorstores import FAISS


def load_pdf(path):
    """
    Loads and parses a PDF file into a list of documents.
    
    Each document typically represents a page or section.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents


def split_documents(documents):
    """
    Splits documents into overlapping chunks.

    Why:
    - LLMs have context size limits
    - Smaller chunks improve retrieval accuracy
    - Overlap preserves semantic continuity between chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Larger chunks work better for technical documents
        chunk_overlap=200     # Overlap ensures important context isn't lost between chunks
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    """
    Converts text chunks into embeddings and stores them in a vector database.

    This enables semantic search instead of keyword matching.
    """
    embeddings = OllamaEmbeddings(model="llama3")

    # FAISS is an in-memory vector store optimized for similarity search
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def create_llm():
    """
    Initializes the language model used for answering questions.

    Using a local model via Ollama keeps the system fully offline.
    """
    llm = OllamaLLM(model="llama3")
    return llm


def ask_question(vector_store, llm):
    """
    Main interaction loop.

    Flow:
    1. Receive user query
    2. Retrieve relevant chunks via semantic search
    3. Build context
    4. Send to LLM
    5. Return answer
    """
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # Retrieve top-k most relevant chunks
        docs = vector_store.similarity_search(query, k=6)

        # ------------------------------
        # DEBUG: Show retrieved chunks
        # ------------------------------
        print("\n[DEBUG] Retrieved context chunks:")
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:300])  # limit output for readability

        # Combine retrieved chunks into a single context string
        context = "\n".join([doc.page_content for doc in docs])

        # Prompt engineering: enforce grounded answers
        prompt = f"""
You are an expert AI assistant.

Answer the question based ONLY on the context below.
If the context is insufficient, say "I don't know".

Be precise and explain clearly.

Context:
{context}

Question:
{query}

Answer:
"""

        # Invoke LLM
        response = llm.invoke(prompt)

        print("\nAnswer:", response)


def main():
    """
    Orchestrates the full pipeline:
    PDF → Chunking → Embeddings → Retrieval → LLM
    """
    print("Loading PDF...")
    documents = load_pdf("pdf/sample.pdf")

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("Loading LLM...")
    llm = create_llm()

    print("\nReady! You can now ask questions about the PDF.")
    ask_question(vector_store, llm)


if __name__ == "__main__":
    main()