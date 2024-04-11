FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./main.py .

RUN mkdir /app/.streamlit && touch /app/.streamlit/secrets.toml
