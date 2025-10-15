from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import PyPDF2

# PostgreSQL connection
import psycopg2

from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ OLD (OpenAI imports) ------------------
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# ----------------------------------------------------------

# ------------------ NEW (Ollama imports) ------------------
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
# ----------------------------------------------------------

from langchain.chains import RetrievalQA
from langchain_community.vectorstores.pgvector import PGVector

app = FastAPI(title="RAG POC API with Document Management")

# ------------------ PostgreSQL connection ------------------
conn = psycopg2.connect(
    dbname="rag_poc",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)
cur = conn.cursor()

# ==========================================================
# OLD: Using OpenAI embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# ==========================================================

# ✅ NEW: Using Ollama embeddings locally
# Make sure to pull the model first: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ------------------ PGVector setup ------------------
CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5433/rag_poc"
TABLE_NAME = "documents"

try:
    vectorstore = PGVector.from_existing_index(
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        table_name=TABLE_NAME,
    )
    print("✅ Loaded existing PGVector index.")
except Exception as e:
    print(f"⚠️ Could not find existing index. Creating new one... ({e})")
    vectorstore = PGVector.from_documents(
        documents=[],  # empty for now; ingestion will add later
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        table_name=TABLE_NAME,
    )
    print("✅ Created new PGVector table.")

retriever = vectorstore.as_retriever()

# ==========================================================
# OLD: Using OpenAI chat model
# llm = ChatOpenAI(model_name="phi3:mini", temperature=0, openai_api_base="http://localhost:11434")
# ==========================================================

# ✅ NEW: Using Ollama locally as the LLM
# Make sure to pull: ollama pull phi3:mini
llm = Ollama(model="phi3:mini", temperature=0)

# ------------------ Retrieval QA chain ------------------
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ------------------ Text splitter & folder setup ------------------
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ---------------- API endpoints ---------------- #
@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(data_folder, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    reader = PyPDF2.PdfReader(file_location)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunks = text_splitter.split_text(text)

    # ✅ Add chunks directly to PGVector via LangChain
    vectorstore.add_texts(chunks)

    return {"message": f"Ingested {file.filename} with {len(chunks)} chunks"}

@app.get("/query")
async def query_rag(q: str):
    answer = qa.run(q)
    return {"query": q, "answer": answer}

@app.get("/list_documents")
async def list_documents():
    cur.execute("SELECT id, content FROM documents LIMIT 10")
    rows = cur.fetchall()
    return {"documents": [{"id": r[0], "content": r[1][:200]+"..."} for r in rows]}

@app.delete("/delete_document/{doc_id}")
async def delete_document(doc_id: int):
    cur.execute("SELECT id FROM documents WHERE id=%s", (doc_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    cur.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
    conn.commit()
    return {"message": f"Deleted document with id {doc_id}"}

@app.delete("/clear_documents")
async def clear_documents():
    cur.execute("DELETE FROM documents")
    conn.commit()
    return {"message": "All documents cleared"}

@app.on_event("shutdown")
def shutdown_event():
    cur.close()
    conn.close()
