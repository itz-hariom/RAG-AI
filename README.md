# RAG-AI

This project is a PDF Question Answering system. It allows us to ask questions from usr own PDF files and get answers that are based only on the content inside those PDFs.

The idea behind this project is to use a technique called RAG (Retrieval Augmented Generation). In simple words, instead of directly asking the AI model, we first search inside the PDFs, find the most relevant parts, and then ask the AI to answer from those specific parts only. This makes the answers more accurate and reduces hallucinations.

Here is how the system works:

First, we put all usr PDF files inside the data folder.

When we run the project, the code extracts text from each PDF using pdfplumber.

The extracted text is cleaned and then split into small pieces called chunks. This is needed so we can store them and search them easily.

Each text chunk is converted into an embedding (a numerical vector) using the MiniLM model from Sentence Transformers.

All these vectors along with the text are stored in a local database called ChromaDB. This works like the “memory” of the system.

When we ask a question, usr question is also converted into a vector in the same way.

The system then searches the vector database and finds the chunks that are most similar to usr question.

These relevant chunks are combined and sent, along with usr question, to the Groq Llama 3 model. This model generates the final answer strictly based on the provided PDF content.

The answer is displayed in a simple Streamlit interface.

The main files in the project are:

rag_pipeline.py — This is the core logic. It handles text extraction, cleaning, chunking, embeddings, vector database creation, retrieval, and sending the final prompt to the LLM.

loaders.py — This file contains a simple function that reads text from PDF files.

app.py — This is the Streamlit application. It creates the user interface where we can ask questions and rebuild the vector database.

data/ — This folder contains the PDF files that us want the system to read.

chroma_db/ — This folder stores the generated embeddings and metadata.

.env — This file contains usr Groq API key.

requirements.txt — This file has all the libraries required to run the project.

To run the project, us need to install the required Python packages, add usr Groq API key in the .env file, place usr PDFs inside the data folder, build the vector database, and finally run the Streamlit app using the command:

python -m streamlit run app.py

After this, us can type any question related to usr PDFs and the system will answer it by actually looking into usr documents.

This project is completely free to run because it uses free embeddings and the free Groq Llama 3 model. It can be used for summarizing notes, searching inside books, understanding assignments, or creating usr own personal document search engine.
