FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core.py .
COPY main.py .
COPY runner.py .
COPY foreclosure.py .
COPY probate.py .
COPY divorce.py .

CMD ["python", "runner.py"]
