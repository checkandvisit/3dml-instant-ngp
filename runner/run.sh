#/bin/bash

DIR=$(dirname ${0})
ROOT_DIR="$( cd ${DIR} >/dev/null 2>&1 && pwd )"
cd $ROOT_DIR/..

scene=kitchen

python3 scripts/run.py --scene data/${scene}/transform.json            \
        --mode nerf                                                         \
        --network base.json                                                 \
        --gui