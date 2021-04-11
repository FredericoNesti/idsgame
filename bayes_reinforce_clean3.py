import os
import gym
import sys
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.bayesian_methods.policy_gradient.reinforce.bayes_reinforce_bestprior2 import BayesReinforceAgent
from util import util

import argparse
import json

parser = argparse.ArgumentParser(description='Dataset')

parser.add_argument('--id_seed', type=int, default=999999991)

parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--lr_decay_rate', type=float, default=0.999)
parser.add_argument('--M', type=int, default=100)
parser.add_argument('--prior_type', type=int, default=1)
parser.add_argument('--env_name', type=str, default="idsgame-random_defense-v0")
parser.add_argument('--input_dim_attacker', type=int, default=33)
parser.add_argument('--output_dim_attacker', type=int, default=30)
parser.add_argument('--prior_state', type=bool, default=True)
parser.add_argument('--alpha_attacker', type=float, default=0.0001)
parser.add_argument('--eval_episodes', type=int, default=100)
parser.add_argument('--num_episodes', type=int, default=10001)
parser.add_argument('--eval_frequency', type=int, default=10)
parser.add_argument('--train_log_frequency', type=int, default=1)
parser.add_argument('--eval_log_frequency', type=int, default=1)

args = parser.parse_args()


def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir

# Program entrypoint
if __name__ == '__main__':
    random_seed = args.id_seed

    with open(default_output_dir() + "/results/" + str(random_seed) + "_args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    util.create_artefact_dirs(default_output_dir(), random_seed)
    # these parameter are changed
    pg_agent_config = PolicyGradientAgentConfig(gamma=1, alpha_attacker=args.alpha_attacker, epsilon=1, render=False, eval_epsilon=0.,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=args.eval_episodes, train_log_frequency=args.train_log_frequency,
                                                epsilon_decay=0., video=False, eval_log_frequency=args.eval_log_frequency,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos/" + str(random_seed),
                                                num_episodes=args.num_episodes,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs/" + str(random_seed),
                                                eval_frequency=args.eval_frequency, attacker=True, defender=False,
                                                video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data/" + str(random_seed),
                                                checkpoint_freq=5000, input_dim_attacker=args.input_dim_attacker, output_dim_attacker=args.output_dim_attacker,
                                                hidden_dim=64,
                                                num_hidden_layers=1, batch_size=args.batchsize,
                                                gpu=True, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard/" + str(random_seed),
                                                optimizer="Adam", lr_exp_decay=True, lr_decay_rate=args.lr_decay_rate)
    env_name = args.env_name
    env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + str(random_seed))
    attacker_agent = BayesReinforceAgent(prior_state=args.prior_state, prior_type=args.prior_type, M=args.M, env=env, config=pg_agent_config)
    attacker_agent.train()
    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result