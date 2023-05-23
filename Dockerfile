FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt update && apt install -y wget

# Prepare the global python environment
RUN apt install -y python3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python
RUN apt install -y python3-pip


ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# for opencv
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install other tools
RUN apt install -y \
    git \
    libsm6  \
    libxext-dev \
    libxrender1 \
    unzip \
    cmake \
    libxml2 libxml2-dev libxslt1-dev \
    dirmngr gnupg2 lsb-release \
    xvfb kmod swig patchelf \
    libopenmpi-dev  libcups2-dev \
    libssl-dev  libosmesa6-dev \
    mesa-utils

# Clean up to make the resulting image smaller
RUN  rm -rf /var/lib/apt/lists/*


# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


COPY ./environment.yml /tmp/
RUN conda env update --file /tmp/environment.yml && \
    conda clean -ya

# Need to install torch and torchvision separately. Their versions are slightly
# higher than the author provided ones.
RUN conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
