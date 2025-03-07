#!/usr/bin/env bash

date

###########################
#		settings
###########################

DATASET="${1:-univ}"
WEIGHT="${2:-50.0}"
SEED="${3:-123}"

echo 'DATASET: '${DATASET}
echo 'WEIGHT: '${WEIGHT}
echo 'SEED: '${SEED}

MODEL=vel
EPOCH=300

PREFIX=experiments/pedestrians/models
METHOD='social'

###########################
#		python
###########################

python trajectron/train.py --data_dir experiments/processed --train_data_dict ${DATASET}_train.pkl --eval_data_dict ${DATASET}_val.pkl --offline_scene_graph yes --train_epochs ${EPOCH} --log_dir ${PREFIX} --log_tag _${DATASET}_${MODEL} --conf ${PREFIX}/${DATASET}_vel/config.json --augment --max_num_neighbors 30 --contrastive_weight ${WEIGHT} --preprocess_workers 12 --contrastive_sampling ${METHOD} --seed ${SEED}

date