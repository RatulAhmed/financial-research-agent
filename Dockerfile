FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY preload.py .
RUN python preload.py

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]