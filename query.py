from langchain.vectorstores import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings

# TODO: Replace with your actual PostgreSQL credentials
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PGVector(
    table_name="documents",
    embedding_function=embeddings,
    connection_string="postgresql://postgres:postgres@localhost:5433/rag_poc"
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model_name="phi3:mini", temperature=0, openai_api_base="http://localhost:11434")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print("\nAnswer:\n", answer)
