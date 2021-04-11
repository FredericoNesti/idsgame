#!/bin/bash
python3 bayes_reinforce_clean3.py \
--id_seed 1209999911 \
--batchsize 4 \
--lr_decay_rate 0.995 \
--M 70 \
--prior_type 1 \
--env_name "idsgame-minimal_defense-v2" \
--input_dim_attacker 44 \
--output_dim_attacker 40 \
--prior_state True \
--alpha_attacker 0.001 \
--eval_episodes 100 \
--num_episodes 10001 \
--eval_frequency 150 \
--train_log_frequency 50 \
--eval_log_frequency 150
