#!/bin/bash

# NO PECP | Author's Dataformat | Author's Checkpoints
python eval_legacy_all_scenes.py --all_objects --model_dir_root model/paper_checkpoints --data_path /data/Datasets/ContourPose/


# PECP | Author's Dataformat | Author's Checkpoints
python eval_legacy_all_scenes.py --all_objects --model_dir_root model/paper_checkpoints --data_path /data/Datasets/ContourPose/ --pecp

