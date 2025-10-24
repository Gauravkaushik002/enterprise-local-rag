import logging
import os
import sys
import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.pgvector import PGVector


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


# ------------------ RAG Query Class ------------------
class RAGQuery:
    """
    Handles Retrieval-Augmented Generation (RAG) queries
    using local Ollama models and PGVector for vector storage.
    """

    def __init__(self,
                 pg_connection_string: str = "postgresql://postgres:postgres@localhost:5433/rag_poc",
                 table_name: str = "documents",
                 llm_model: str = "phi3:mini",
                 embedding_model: str = "nomic-embed-text:latest"):

        self.logger = create_logger("RAG_QUERY", "query.log")
        self.logger.info("Initializing RAGQuery class...")

        self.pg_connection_string = pg_connection_string
        self.table_name = table_name

        # ----------------- Embeddings -----------------
        try:
            self.logger.info("Loading embeddings model: %s", embedding_model)
            self.embeddings = OllamaEmbeddings(model=embedding_model)
        except Exception as e:
            self.logger.exception("Failed to initialize embeddings: %s", e)
            raise

        # ----------------- PGVector Initialization -----------------
        try:
            self.logger.info("Connecting to PGVector table: %s", self.table_name)
            self.vectorstore = PGVector.from_existing_index(
                embedding=self.embeddings,
                connection_string=self.pg_connection_string,
                table_name=self.table_name
            )
            self.logger.info("Loaded existing PGVector index.")
        except Exception as e:
            self.logger.warning("No existing PGVector index found (%s). Creating a new one...", e)
            self.vectorstore = PGVector.from_documents(
                documents=[],
                embedding=self.embeddings,
                connection_string=self.pg_connection_string,
                table_name=self.table_name
            )
            self.logger.info("Created new PGVector table.")

        # ----------------- Retriever & LLM -----------------
        try:
            self.retriever = self.vectorstore.as_retriever()
            self.llm = Ollama(model=llm_model, temperature=0)
            self.logger.info("Loaded LLM model: %s", llm_model)
        except Exception as e:
            self.logger.exception("Failed to initialize LLM or retriever: %s", e)
            raise

        # ----------------- QA Chain -----------------
        try:
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff"
            )
            self.logger.info("RetrievalQA chain initialized successfully.")
        except Exception as e:
            self.logger.exception("Failed to create RetrievalQA chain: %s", e)
            raise

    # ------------------ Enhanced run_query with Step Logs ------------------
    def run_query(self, query: str) -> str:
        """Run a RAG query and return the answer with detailed step logs."""
        self.logger.info("Received query: %s", query)
        overall_start = time.time()

        try:
            # Step 1: Document Retrieval
            self.logger.info("Step 1: Retrieving relevant documents...")
            retrieval_start = time.time()
            docs = self.retriever.get_relevant_documents(query)
            retrieval_time = time.time() - retrieval_start
            self.logger.info("Step 1 Completed: Retrieved %d documents in %.2f seconds.",
                             len(docs), retrieval_time)

            # Optional: Log first few doc snippets for visibility
            for i, d in enumerate(docs[:3]):
                text_preview = d.page_content[:200].replace("\n", " ")
                self.logger.info("Doc %d preview: %s...", i + 1, text_preview)

            # Step 2: LLM Answer Generation
            self.logger.info("Step 2: Generating response using LLM...")
            generation_start = time.time()
            answer = self.qa.combine_documents_chain.run(
                {"input_documents": docs, "question": query}
            )
            generation_time = time.time() - generation_start
            self.logger.info("Step 2 Completed: LLM generation finished in %.2f seconds.", generation_time)

            total_time = time.time() - overall_start
            self.logger.info("✅ Query executed successfully in %.2f seconds (Retrieval: %.2fs | Generation: %.2fs)",
                             total_time, retrieval_time, generation_time)

            return answer

        except Exception as e:
            self.logger.exception("Error during query execution: %s", e)
            raise

    # ------------------ Retriever Refresh ------------------
    def refresh_retriever(self):
        """Refresh the retriever to include newly ingested documents."""
        self.logger.info("Refreshing retriever to include new documents...")
        try:
            self.vectorstore = PGVector.from_existing_index(
                embedding=self.embeddings,
                connection_string=self.pg_connection_string,
                table_name=self.table_name
            )
            self.retriever = self.vectorstore.as_retriever()
            self.qa.retriever = self.retriever
            self.logger.info("Retriever refreshed successfully.")
        except Exception as e:
            self.logger.warning("Failed to refresh retriever (%s). Creating a new one...", e)
            self.vectorstore = PGVector.from_documents(
                documents=[],
                embedding=self.embeddings,
                connection_string=self.pg_connection_string,
                table_name=self.table_name
            )
            self.retriever = self.vectorstore.as_retriever()
            self.qa.retriever = self.retriever
            self.logger.info("New vectorstore created during refresh.")


# ------------------ Standalone CLI Usage ------------------
if __name__ == "__main__":
    rag_query = RAGQuery()
    rag_query.logger.info("RAG Query CLI started. Type 'exit' to quit.")

    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() == "exit":
            rag_query.logger.info("Exiting RAG Query CLI.")
            break
        try:
            answer = rag_query.run_query(user_input)
            print("\nAnswer:\n", answer)
        except Exception as e:
            rag_query.logger.error("Query failed: %s", e)
