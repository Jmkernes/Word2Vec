#!/bin/bash

echo "===== Setting up configuration ====="

WINDOW_SIZE=11
TEXT_FILE=train.txt
BATCH_SIZE=128
NUM_NS=7
EPOCHS=1
THRESHOLD=5e-5
D_MODEL=128
LEARNING_RATE=1e-3
BUFFER_SIZE=10000
ANNEAL_THRESHOLD=False
WEIGHTS_DIRECTORY=embedding_weights

echo "===== Beginning training ====="

python3 train.py \
	--window_size=${WINDOW_SIZE} \
	--text_file=${TEXT_FILE} \
	--batch_size=${BATCH_SIZE} \
	--num_ns=${NUM_NS} \
	--epochs=${EPOCHS} \
	--threshold=${THRESHOLD} \
	--d_model=${D_MODEL} \
	--learning_rate=${LEARNING_RATE} \
	--buffer_size=${BUFFER_SIZE} \
	--anneal_threshold=${ANNEAL_THRESHOLD} \
	--weights_directory=${WEIGHTS_DIRECTORY}
