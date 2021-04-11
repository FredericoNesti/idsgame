#!/bin/bash
python3 bayes_reinforce_clean.py \
--id_seed 1208888800 \
--batchsize 4 \
--lr_decay_rate 0.995 \
--M 70 \
--env_name "idsgame-minimal_defense-v2" \
--input_dim_attacker 44 \
--output_dim_attacker 40 \
--alpha_attacker 0.001 \
--eval_episodes 100 \
--num_episodes 10001 \
--eval_frequency 150 \
--train_log_frequency 50 \
--eval_log_frequency 150 \
--nu 0.0005 \
--gpu True \
--attacker True 
