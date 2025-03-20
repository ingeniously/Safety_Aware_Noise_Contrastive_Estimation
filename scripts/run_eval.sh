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

cd experiments/Agents

MODEL=vel
PREFIX=models
FOLDERNAME=${PREFIX}/snce_${DATASET}_${MODEL}
OPTDIR=results

###########################
#		python
###########################

for CKPT in {50..300..10}
do
	python evaluate.py --model ${FOLDERNAME} --checkpoint ${CKPT} --data ../processed/${DATASET}_test.pkl --output_path ${OPTDIR} --output_tag ${DATASET}_${MODE}_12 --node_type Agent --contrastive_weight ${WEIGHT} --seed ${SEED} || break
done

date