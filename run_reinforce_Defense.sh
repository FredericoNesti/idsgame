#!/bin/bash
python3 reinforce.py \
--id_seed 510800 \
--batchsize 4 \
--lr_decay_rate 0.995 \
--env_name "idsgame-random_attack-v1" \
--input_dim_defender 33 \
--output_dim_defender 33 \
--alpha_defender 0.001 \
--eval_episodes 100 \
--num_episodes 10001 \
--eval_frequency 150 \
--train_log_frequency 50 \
--eval_log_frequency 150 \
--defender True \
--gpu False 
