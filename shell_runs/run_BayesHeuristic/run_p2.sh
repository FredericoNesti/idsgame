#!/bin/bash
python3 ../../bayes_prior_reinforce.py --seed 11 --prior_type 5 --gpu True --batchsize 1 --experiment_id "BayesHeuristic_0_MinDef19_1000_05_000000333" --entropy_reg 0.001 --lr_decay_rate 0.9 --epsilon 0.5 --alpha_attacker 0.000001 --epsilon_decay 0.1 --min_epsilon 0.01 --M 100
