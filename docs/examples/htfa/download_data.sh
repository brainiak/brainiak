#!/bin/bash

if [ -d "data/" ]; then
    echo "Skipping download of data for HTFA notebook, already present"
else
    mkdir data
    wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate -q \
        "https://docs.google.com/uc?export=download&id=1IBA39ZZjeGS1u_DvZdiw1AZZQMS3K5q0" -O- \
        | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p' > confirm
    wget --load-cookies cookies.txt --no-check-certificate -q \
        "https://docs.google.com/uc?export=download&confirm="$(cat confirm)"&id=1IBA39ZZjeGS1u_DvZdiw1AZZQMS3K5q0" -O data/pieman.zip
    rm cookies.txt confirm
    unzip data/pieman.zip -d data/
    rm data/pieman.zip
fi

