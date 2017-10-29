FROM ubuntu:16.04

# Group 1 must be synced with README
# Group 2 must be synced with requirements for examples
# Group 3 must be synced with rest of Dockerfile
# Group 4 is optional
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libmpich-dev \
    mpich \
    python3-dev \
    python3-pip \
    python3-venv \
    \
    curl \
    unzip \
    wget \
    \
    screen \
    \
    less \
    man \
    vim

WORKDIR /mnt

RUN set -e \
    && python3 -m pip install --user -U pip \
    && python3 -m pip install -U brainiak \
    && python3 -m pip download --no-deps --no-binary :all: brainiak \
    && export BRAINIAK_VERSION=$(basename brainiak-* .tar.gz | cut -b 10-) \
    && tar -xf brainiak-*.tar.gz \
    && for example in brainiak-$BRAINIAK_VERSION/examples/*/requirements.txt; \
        do python3 -m pip install --user -U -r $example ; done

RUN echo PATH=\"\$HOME/.local/bin:\$PATH\" >> $HOME/.profile \
    && echo "shell -bash" >> ~/.screenrc

EXPOSE 8888

ENTRYPOINT ["/bin/bash", "-l"]
