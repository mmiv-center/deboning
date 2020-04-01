FROM ubuntu:18.04

RUN apt update -y && apt upgrade -qy && apt install wget git -y \
    && cd /tmp/ && wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh \
    && bash Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/conda

ENV PATH="/opt/conda/bin:$PATH"

RUN conda shell.bash hook && conda config --set auto_activate_base false \
    && conda init bash && conda update -n base -c defaults conda && . ~/.bashrc \
    && conda create --name fastai \
    && conda activate fastai && conda install -c pytorch -c fastai fastai \
    && conda install -c conda-forge pydicom torchvision

RUN cd /root/ && git clone https://github.com/mmiv-center/deboning \
    && cd deboning