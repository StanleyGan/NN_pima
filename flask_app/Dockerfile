FROM python:3.6-slim
COPY ./simple_pima_app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./model.pkl /deploy/
WORKDIR /deploy/
RUN pip3 install -r requirements.txt
EXPOSE 8888
ENTRYPOINT ["python3", "simple_pima_app.py"]
