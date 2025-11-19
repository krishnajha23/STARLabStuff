#!/bin/bash

export MUJOCO_GL=egl

cd ~/CIMER/mj_envs
pip install -e .
cd ~/CIMER
bash remove.sh

# Don't use LD_PRELOAD with EGL - it conflicts
python3 hand_dapg/dapg/controller_training/visualize.py \
  --eval_data Samples/Hammer/Hammer_task.pickle \
  --visualize False \
  --save_fig True \
  --config Samples/Hammer/CIMER/job_config.json \
  --policy Samples/Hammer/CIMER/best_eval_sr_policy.pickle \
  --record_video True \
  --save_frames True