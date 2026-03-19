FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python preload.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]