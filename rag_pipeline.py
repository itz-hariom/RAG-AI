# rag_pipeline.py
import os
import re
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from loaders import load_pdf
from groq import Groq


load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# eMBEDDING CUSTOMING CLAS
class Transforming(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        #Give lists of text
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()


# I'm cleaning the text here
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Building the vector
def build_vector_db():
    docs = []

    # 1. Load all PDFs from /data
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            print(f"Loading PDF: {path}")

            text = load_pdf(path)
            text = clean_text(text)

            docs.append(Document(page_content=text, metadata={"source": file}))

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print("Total chunks:", len(chunks))

    # 3. Initialize embedding class
    embedding_function = Transforming()

    # 4. Init Chroma DB
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function
    )

    # 5. Add chunks manually
    texts = [doc.page_content for doc in chunks]
    metas = [doc.metadata for doc in chunks]

    vectordb.add_texts(texts=texts, metadatas=metas)
    vectordb.persist()

    print("Vector DB created successfully!")
    return vectordb


# Loading vector
def load_vector_db():
    embedding_function = Transforming()

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function
    )

    return vectordb


# GROQ LLM ko free kar rahe h
def groq_llm(prompt: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# RAG Query pipeline
def answer_query(query: str):
    vectordb = load_vector_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"SOURCE: {d.metadata['source']}\n{d.page_content}"
        for d in docs
    )

    prompt = f"""
Use ONLY this context to answer.
If the answer is not present, say "Not found in the PDF."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    return groq_llm(prompt)


# main
if __name__ == "__main__":
    build_vector_db()
