
FROM python:3.11.1

WORKDIR ./
RUN apt-get update
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
RUN mkdir -p ./vector_db
ADD ./vector_db ./vector_db
COPY ./app.py ./app.py

CMD ["streamlit", "run", "app.py"]