FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
# Group 1 must be synced with README
# Group 2 must be synced with requirements for examples
# Group 3 must be synced with rest of Dockerfile
# Group 4 is optional
# Group 5 must be synced with requirements for tutorials
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libmpich-dev \
    mpich \
    python3-dev \
    python3-pip \
    python3-venv \
    \
    python3-tk\
    ffmpeg\
    curl \
    unzip \
    wget \
    \
    screen \
    \
    less \
    man \
    vim \
    \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY brainiak-* /mnt/brainiak

WORKDIR /mnt/brainiak

COPY tutorials/tutorials tutorials

RUN set -e \
    && python3 -m pip install --user -U pip \
    && python3 -m pip install --user -U -r tutorials/requirements.txt \
    && for example in examples/*/requirements.txt; \
        do python3 -m pip install --user -U -r $example ; done \
    && rm -rf ~/.cache/pip

RUN echo PATH=\"\$HOME/.local/bin:\$PATH\" >> $HOME/.profile \
    && echo "shell -bash" >> ~/.screenrc

EXPOSE 8899

ENTRYPOINT ["/bin/bash", "-l"]
CMD ["tutorials/run_jupyter_docker.sh"]
