#/bin/bash

DIR=$(dirname ${0})
ROOT_DIR="$( cd ${DIR} >/dev/null 2>&1 && pwd )"
cd $ROOT_DIR/..

scene="0223-1120"

# python3 scripts/colmap2nerf.py --video_in data/ours/${scene}/${scene}.mp4   \
#         --run_colmap                                                        \
#         --colmap_db data/ours/${scene}/colmap.db                            \
#         --text data/ours/${scene}/colmap/                                   \
#         --out data/ours/${scene}/transform.json

#python3 scripts/colmap2nerf.py --images data/ours/${scene}/images           \
#        --run_colmap                                                        \
#        --colmap_db data/ours/${scene}/colmap.db                            \
#        --text data/ours/${scene}/colmap/                                   \
#        --out data/ours/${scene}/transform.json

sed -i "s/.\/data\/${scene}\/images/images/g" data/${scene}/transform.json

python3 scripts/run.py --scene data/${scene}/transform.json            \
        --mode nerf                                                         \
        --network base.json                                                 \
        --n_steps 2000                                                      \
        --screenshot_dir data/${scene}/screenshot/                     \
        --screenshot_transforms data/${scene}/transform.json           \
        --save_snapshot data/${scene}/weight.msgpack

# ffmpeg -y -r 2.0 -i data/ours/${scene}/screenshot/%04d.png \
#         -c:v libx264 -vf fps=2.0 data/ours/${scene}/result.mp4

#python3 scripts/run.py --scene data/ours/${scene}/transform.json            \
#        --mode nerf                                                         \
#        --network base.json                                                 \
#        --width 600                                                         \
#        --height 600                                                        \
#        --load_snapshot data/ours/${scene}/weight.msgpack                   \
#        --gui
