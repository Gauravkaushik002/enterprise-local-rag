import os
import PyPDF2
# TODO: Install psycopg2 and set actual PostgreSQL credentials
import psycopg2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ OLD (OpenAI imports) ------------------
# from langchain.embeddings import OpenAIEmbeddings
# ----------------------------------------------------------

# ------------------ NEW (Ollama imports) ------------------
from langchain_community.embeddings import OllamaEmbeddings
# ----------------------------------------------------------

# ------------------ PostgreSQL connection ------------------
# TODO: Replace with your actual PostgreSQL credentials
conn = psycopg2.connect(
    dbname="rag_poc",
    user="postgres",
    password="postgres",       # ✅ replace YOUR_PASSWORD_HERE
    host="localhost",
    port=5433                  # ✅ use your Docker port (if using pgvector docker)
)
cur = conn.cursor()

# ==========================================================
# OLD: Using OpenAI embeddings (requires internet + API key)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# ==========================================================

# ✅ NEW: Using Ollama embeddings locally (no API key needed)
# Make sure you have pulled the model first:
#   ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ------------------ Folder and text splitter ------------------
data_folder = "data"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ------------------ PDF ingestion and embedding ------------------
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_folder, filename)
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Split into chunks
        chunks = text_splitter.split_text(text)

        # Generate and insert embeddings
        for chunk in chunks:
            # OLD: OpenAI embedding call
            # emb = embeddings.embed_query(chunk)

            # ✅ NEW: Ollama embedding call
            emb = embeddings.embed_query(chunk)

            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (chunk, emb)
            )

        print(f"✅ Ingested {filename} with {len(chunks)} chunks")

# ------------------ Commit and close ------------------
conn.commit()
cur.close()
conn.close()
print("✅ All documents ingested and database connection closed.")
