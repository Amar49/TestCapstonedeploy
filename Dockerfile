FROM python:3.11.4
COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
COPY . ./app
WORKDIR app
ENTRYPOINT python llm_NLPandavas.py
