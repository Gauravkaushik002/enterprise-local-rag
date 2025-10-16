# Project Path
C:/My_Data/Artificial_Intelligence/RAG/RAG_Project_Self_Learning/workspace/rag_poc

# Install dependencies (if not already done):
pip install -r requirements.txt

# Database
C:/My_Data/Artificial_Intelligence/RAG/RAG_Project_Self_Learning/pgvector_docker

docker compose ps                # See if PostgreSQL is running
docker compose logs -f postgres  # Follow logs
docker compose down              # Stop containers (data persists)
docker compose up -d             # Restart again

top all running Docker containers
docker stop $(docker ps -q)
***************Connect via psql (optional)
psql -h localhost -p 5433 -U postgres -d postgres

# Start FastAPI Server (Run the API server using uvicorn):
uvicorn api:app --reload

# By default, FastAPI runs on http://127.0.0.1:8000
You will see output like:
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXX]

# Check API Availability
Open your browser and go to:
http://127.0.0.1:8000/docs

