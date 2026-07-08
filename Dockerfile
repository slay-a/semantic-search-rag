# Streamlit RAG app + pgvector-ready runtime.
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; psycopg[binary] ships its own libpq.
RUN pip install --no-cache-dir --upgrade pip

# Install the package (with the Postgres extra) first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]"

COPY app.py ./
COPY data ./data

ENV RAG_STORE_BACKEND=pgvector \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Streamlit must bind 0.0.0.0 to be reachable from outside the container.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
