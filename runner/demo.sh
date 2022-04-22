#!/bin/bash

REPO_DIR=$(dirname ${0})
workspaceRoot="$( cd ${REPO_DIR} >/dev/null 2>&1 && pwd )"
cd ${workspaceRoot}/..

if [ ! -d "data/nerf_synthetic" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG" -O data/nerf_synthetic.zip && rm -rf /tmp/cookies.txt
    unzip data/nerf_synthetic.zip -d data
fi

python3 scripts/run.py --mode nerf --scene data/nerf_synthetic/lego/transforms_train.json --width 1280 --height 720 --gui