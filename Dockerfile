FROM python:3.6

RUN apt-get update
RUN apt-get install -y htop

ADD requirements.txt /app/
WORKDIR /app

RUN pip3 install -r requirements.txt
RUN pip3 install imageai --upgrade

ADD . /app

CMD python /app/ia.py
