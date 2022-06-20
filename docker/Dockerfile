FROM python:3.10-slim-bullseye
WORKDIR /app
COPY docker/requirements-docker.txt ./requirements-docker.txt
RUN pip3 install -r requirements-docker.txt
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
