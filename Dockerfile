FROM python:3.10

RUN apt -y update && apt-get -y upgrade
RUN apt install -y ffmpeg
RUN apt install -y python3-pip
RUN apt-get -y install python3-tk
RUN /usr/local/bin/python -m pip install --user --upgrade pip setuptools wheel

# Because of requirement Dora==0.0.3
COPY ./requirements.txt ./requirements.txt
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN /usr/local/bin/python -m pip install --user -r ./requirements.txt

# Save this until the end to avoid cache invalidation with each code change
COPY ./ ./

CMD ["bash"]
