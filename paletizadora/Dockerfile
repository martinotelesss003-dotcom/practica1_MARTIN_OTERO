FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CMD ["sh","-c","gunicorn --workers 1 --threads 2 --timeout 90 --bind 0.0.0.0:${PORT:-8000} app:app"]
