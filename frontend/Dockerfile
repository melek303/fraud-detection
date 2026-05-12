FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/
COPY frontend/ ./frontend/
COPY src/__init__.py ./src/
COPY src/pipeline/__init__.py ./src/pipeline/
COPY src/pipeline/preprocess.py ./src/pipeline/


EXPOSE 8501

CMD ["sh", "-c", "streamlit run frontend/streamlit.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]