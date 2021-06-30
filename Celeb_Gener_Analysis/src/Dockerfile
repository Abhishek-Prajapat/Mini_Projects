FROM ubuntu:20.04

RUN apt -y update 
RUN apt install -y python3-dev 
RUN apt -y update 
RUN apt install -y wget
RUN apt -y update

ENV PATH='/root/miniconda3/bin:${PATH}'
ARG PATH='/root/miniconda3/bin:${PATH}'

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh


RUN conda create -y -n ml python=3.8

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate ml \
    && pip install -r requirements_inference.txt"