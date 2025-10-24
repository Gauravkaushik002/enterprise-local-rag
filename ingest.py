import logging
import os
import sys
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# ------------------ Logging Setup ------------------
LOG_FOLDER = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)

def create_logger(name: str, filename: str) -> logging.Logger:
    log_file = os.path.join(LOG_FOLDER, filename)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.hasHandlers():
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# ------------------ PDF Ingestor Class ------------------
class PDFIngestor:
    def __init__(self, conn, cur, data_folder="data"):
        self.conn = conn
        self.cur = cur
        self.data_folder = data_folder
        self.logger = create_logger("PDF_INGEST", "ingest.log")

        os.makedirs(self.data_folder, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.logger.info("PDFIngestor initialized. Watching folder: %s", self.data_folder)

    def ingest_pdf_file(self, file_path):
        try:
            filename = os.path.basename(file_path)
            self.logger.info("Processing PDF: %s", filename)

            # Extract text
            reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if not text.strip():
                self.logger.warning("No text found in PDF: %s", filename)
                return [], 0

            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            self.logger.info("Split PDF into %d chunks.", len(chunks))

            # Insert embeddings
            success_count = 0
            for chunk in chunks:
                try:
                    emb = self.embeddings.embed_query(chunk)
                    self.cur.execute(
                        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                        (chunk, emb)
                    )
                    success_count += 1
                except Exception as e:
                    self.logger.error("Chunk insert failed: %s", e)
                    self.conn.rollback()  # rollback this transaction so connection stays usable

            self.conn.commit()
            self.logger.info("✅ Successfully inserted %d chunks for PDF '%s'.", success_count, filename)
            return chunks, success_count

        except Exception as e:
            self.logger.exception("Failed to ingest PDF '%s': %s", file_path, e)
            self.conn.rollback()
            raise
