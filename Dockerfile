FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main_page.py", "--server.port=8501", "--server.address=0.0.0.0"]
