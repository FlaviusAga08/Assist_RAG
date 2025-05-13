import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)
from langchain_chroma import Chroma


# Set API Key securely
os.environ["OPENAI_API_KEY"] = "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"

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str
    sources: List[str]

# Load documents from directory
doc_directory = "/home/flavius/Documents/Sistem RAG Licitatii publice/FAQ pentru AI/Documente utile"
persist_directory = "db"

all_docs = []

# Load documents and add metadata
for path in Path(doc_directory).rglob('*'):
    if path.is_file():
        filename = path.name
        file_path = os.path.join(doc_directory, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(file_path)
            elif filename.endswith(".docx") or filename.endswith(".doc"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif filename.lower().endswith((".txt", ".md", ".rtf")):
                loader = UnstructuredFileLoader(file_path)
            else:
                print(f"Skipped unsupported file: {filename}")
                continue

            docs = loader.load()

            # Inject metadata
            for doc in docs:
                doc.metadata["source"] = filename  # or file_path if you want full path

            all_docs.extend(docs)

        except Exception as e:
            print(f"Error loading {filename}: {e}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(all_docs)

# Embedding setup
embedding = OpenAIEmbeddings()

# Vector DB setup
# Update Chroma initialization and remove the direct embedding_function argument
vectordb = Chroma.from_documents(
    texts, 
    embedding, 
    persist_directory=persist_directory
)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Initialize language model (OpenAI)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Setup the retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    try:
        response = qa_chain.invoke(request.query)

        result_text = response["result"]
        #sources = [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]]
        #sources = list({doc.metadata.get("source", "Unknown") for doc in response["source_documents"]})
        sources = list({
                        f'{doc.metadata.get("source", "Unknown")} - page {doc.metadata.get("page", "N/A")+1}'
                            for doc in response["source_documents"]
            })

        return QueryResponse(result=result_text, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
