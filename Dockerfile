FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY data/ ./data/

CMD ["python", "app/app.py"]
