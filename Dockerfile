FROM ubuntu:latest
RUN apt-get update -y

RUN apt-get -y install python3-pip
RUN pip3 install --upgrade setuptools

# create model_user
RUN useradd --create-home -u 1008 model_user

# copying folder to Docker container
COPY app ./app
COPY config.yml ./config.yml
COPY mlserving.py ./mlserving.py
COPY model_functions.py ./model_functions.py
COPY model_handler.py ./model_handler.py
COPY requirements.txt ./requirements.txt
COPY start.sh ./start.sh
COPY test.sh ./test.sh

# starting container
ENTRYPOINT [ "/bin/bash" ]
CMD ["start.sh"]