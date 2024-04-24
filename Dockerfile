FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt .

RUN apt update && apt install gcc -y

RUN pip install --no-cache-dir -r requirements.txt

COPY ./main.py .

COPY ./database.py .

RUN mkdir /app/.streamlit && touch /app/.streamlit/secrets.toml
