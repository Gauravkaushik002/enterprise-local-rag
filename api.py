import os
import logging
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
import psycopg2
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.pgvector import PGVector
from ingest import PDFIngestor

# ------------------ Logging Setup ------------------
LOG_FOLDER = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)  # Ensure logs folder exists

def create_logger(name: str, filename: str) -> logging.Logger:
    log_file = os.path.join(LOG_FOLDER, filename)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent uvicorn from overriding

    if not logger.hasHandlers():
        # File handler
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

# ------------------ Global API Logger ------------------
api_logger = create_logger("RAG_POC", "api.log")
api_logger.info("API logger initialized.")

# ------------------ RAG POC Class ------------------
class RAGPOC:
    def __init__(self):
        self.logger = api_logger
        self.logger.info("Initializing RAG POC API.")

        # FastAPI app
        self.app = FastAPI(title="RAG POC API")

        # PostgreSQL connection
        self.conn = psycopg2.connect(
            dbname="rag_poc",
            user="postgres",
            password="postgres",
            host="localhost",
            port=5433
        )
        self.cur = self.conn.cursor()
        self.logger.info("Connected to PostgreSQL.")

        # Embeddings & PGVector
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = self._init_pgvector()
        self.retriever = self.vectorstore.as_retriever()

        # LLM & RetrievalQA
        self.llm = Ollama(model="phi3:mini", temperature=0)
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever, chain_type="stuff")

        # PDF Ingestor
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)
        self.ingestor = PDFIngestor(self.conn, self.cur, data_folder=self.data_folder)

        # Setup API routes
        self._setup_routes()
        self.app.on_event("shutdown")(self._shutdown_event)

    def _init_pgvector(self):
        CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5433/rag_poc"
        TABLE_NAME = "documents"
        try:
            vs = PGVector.from_existing_index(
                embedding=self.embeddings,
                connection_string=CONNECTION_STRING,
                table_name=TABLE_NAME
            )
            self.logger.info("Loaded existing PGVector index.")
        except Exception as e:
            self.logger.warning("No existing PGVector index. Creating new one. (%s)", e)
            vs = PGVector.from_documents(
                documents=[],
                embedding=self.embeddings,
                connection_string=CONNECTION_STRING,
                table_name=TABLE_NAME
            )
            self.logger.info("Created new PGVector table.")
        return vs

    def _setup_routes(self):
        @self.app.post("/ingest_pdf")
        async def ingest_pdf(file: UploadFile = File(...)):
            filename = file.filename or "uploaded.pdf"
            file_path = os.path.join(self.ingestor.data_folder, filename)
            os.makedirs(self.ingestor.data_folder, exist_ok=True)

            try:
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                self.logger.info("Saved uploaded PDF: %s", file_path)

                chunks, num_chunks = self.ingestor.ingest_pdf_file(file_path)
                if num_chunks > 0:
                    self.vectorstore.add_texts(chunks)

                return {"message": f"Ingested {filename} with {num_chunks} chunks"}

            except Exception as e:
                self.logger.exception("Error ingesting PDF.")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/query")
        async def query_rag(q: str):
            try:
                answer = self.qa.run(q)
                return {"query": q, "answer": answer}
            except Exception as e:
                self.logger.exception("Error during query.")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/list_documents")
        async def list_documents():
            try:
                self.cur.execute("SELECT id, content FROM documents LIMIT 10")
                rows = self.cur.fetchall()
                return {"documents": [{"id": r[0], "content": r[1][:200]+"..."} for r in rows]}
            except Exception as e:
                self.logger.exception("Error fetching documents.")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/delete_document/{doc_id}")
        async def delete_document(doc_id: int):
            try:
                self.cur.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
                self.conn.commit()
                return {"message": f"Deleted document {doc_id}"}
            except Exception as e:
                self.logger.exception("Error deleting document.")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/clear_documents")
        async def clear_documents():
            try:
                self.cur.execute("DELETE FROM documents")
                self.conn.commit()
                return {"message": "All documents cleared"}
            except Exception as e:
                self.logger.exception("Error clearing documents.")
                raise HTTPException(status_code=500, detail=str(e))

    def _shutdown_event(self):
        self.cur.close()
        self.conn.close()
        self.logger.info("Database connection closed on shutdown.")

# ------------------ Instantiate FastAPI ------------------
rag_poc_instance = RAGPOC()
app = rag_poc_instance.app
