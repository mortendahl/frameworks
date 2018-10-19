FROM ubuntu:16.04
WORKDIR /root

RUN apt-get update && apt-get install -y \
  python3-pip \
  git

RUN pip3 install --upgrade pip

ADD install.sh .
RUN ["bash", "install.sh"]

ADD config.json
add README.md .
add source/* ./Scripts/
