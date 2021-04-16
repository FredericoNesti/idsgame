#!/bin/bash
python3 ../../bayes_prior_reinforce.py --seed 10 --prior_type 4 --gpu True --batchsize 1 --experiment_id "BayesHeuristic_0_MinDef19_1000_04_000000555" --entropy_reg 0.1 --lr_decay_rate 0.9 --epsilon 0.999 --alpha_attacker 0.000001 --epsilon_decay 0.1 --min_epsilon 0.0001 --M 100
