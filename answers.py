import os
import json
import nltk
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from dotenv import load_dotenv
from models import Base, engine, SessionLocal, QueryHistory

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)
from langchain_chroma import Chroma

# Optional Excel loader (requires pandas and openpyxl)
try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    excel_supported = True
except ImportError:
    excel_supported = False
    print(" Excel files will be skipped. Run pip install pandas openpyxl to enable.")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
doc_location = os.getenv("LOCATION")

# Download NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Initialize DB
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
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

# Convert .doc to .docx using LibreOffice
def convert_doc_to_docx(doc_path: str) -> str:
    output_dir = os.path.dirname(doc_path)
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "docx", "--outdir", output_dir, doc_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        converted_path = doc_path.replace(".doc", ".docx")
        if os.path.exists(converted_path):
            print(f" Converted: {doc_path} → {converted_path}")
            return converted_path
    except subprocess.CalledProcessError:
        print(f" Failed to convert: {doc_path}")
    return None

# Load documents
doc_directory = os.getenv("DOC_LOCATION")
persist_directory = "db"

all_docs = []

for path in Path(doc_location).rglob("*"):
    if path.is_file():
        filename = path.name
        file_path = str(path)
        loader = None

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".doc"):
                converted = convert_doc_to_docx(file_path)
                if converted:
                    loader = Docx2txtLoader(converted)
            elif filename.endswith(".xlsx") and excel_supported:
                loader = UnstructuredExcelLoader(file_path)
            elif filename.lower().endswith((".txt", ".md", ".rtf")):
                loader = UnstructuredFileLoader(file_path)

            if loader:
                print(f" Loading: {filename}")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                all_docs.extend(docs)
            else:
                print(f" Skipped unsupported file: {filename}")

        except Exception as e:
            print(f" Error loading {filename}: {e}")

# Split and embed documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(all_docs)

embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Main query endpoint
@app.post("/query", response_model=List[QueryResponse])
async def run_query(request: QueryRequest):
    db = SessionLocal()
    try:
        # 1. Check for exact match in query history
        existing_entry = db.query(QueryHistory).filter(QueryHistory.query == request.query).first()
        if existing_entry:
            return [QueryResponse(
                result=existing_entry.result,
                sources=json.loads(existing_entry.sources)
            )]

        # 2. Run new query with LangChain
        response = qa_chain.invoke(request.query)
        result_text = response["result"]

        sources = []
        if response["source_documents"]:
            doc = response["source_documents"][0]
            page = doc.metadata.get("page", None)
            page_str = f" - page {page + 1}" if isinstance(page, int) else ""
            source = f'{doc.metadata.get("source", "Unknown")}{page_str}'
            sources.append(source)

        # 3. Save result
        history_entry = QueryHistory(
            query=request.query,
            result=result_text,
            sources=json.dumps(sources)
        )
        db.add(history_entry)
        db.commit()

        # 4. Return only current result
        return [QueryResponse(result=result_text, sources=sources)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# History endpoint
@app.get("/history", response_model=List[QueryResponse])
async def get_history():
    db = SessionLocal()
    try:
        history = db.query(QueryHistory).all()
        return [
            QueryResponse(
                result=entry.result,
                sources=json.loads(entry.sources)
            )
            for entry in history
        ]
    finally:
        db.close()
