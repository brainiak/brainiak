#!/bin/bash
find . -mindepth 1 -type d -name 'download_data.sh' -execdir bash download_data.sh \;
