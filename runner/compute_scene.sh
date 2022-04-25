#!/bin/bash

REPO_DIR=$(dirname ${0})
workspaceRoot="$( cd ${REPO_DIR} >/dev/null 2>&1 && pwd )"
cd ${workspaceRoot}/..

python3 -m instant_ngp_3dml compute_scene 0223-1120