
import os
import gym
import sys
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.bayesian_methods.policy_gradient.reinforce.bayes_reinforce_clean2 import BayesReinforceAgent
from util import util

import argparse
import json

parser = argparse.ArgumentParser(description='Bayes version')

parser.add_argument('--id_seed', type=int, default=5)

parser.add_argument('--batchsize', type=int, default=10)
parser.add_argument('--lr_decay_rate', type=float, default=0.999)
parser.add_argument('--M', type=int, default=5)
parser.add_argument('--env_name', type=str, default="idsgame-random_attack-v0")

parser.add_argument('--input_dim_attacker', type=int, default=33)
parser.add_argument('--output_dim_attacker', type=int, default=30)

#parser.add_argument('--input_dim_defender', type=int, default=33)
#parser.add_argument('--output_dim_defender', type=int, default=33)


parser.add_argument('--alpha_attacker', type=float, default=0.0001)
#parser.add_argument('--alpha_defender', type=float, default=0.0001)

parser.add_argument('--eval_episodes', type=int, default=10)
parser.add_argument('--num_episodes', type=int, default=101)
parser.add_argument('--eval_frequency', type=int, default=10)
parser.add_argument('--train_log_frequency', type=int, default=1)
parser.add_argument('--eval_log_frequency', type=int, default=1)

parser.add_argument('--risk_averse', type=bool, default=False)
parser.add_argument('--static', type=bool, default=False)

#parser.add_argument('--defender', type=bool, default=False)
parser.add_argument('--attacker', type=bool, default=True)

parser.add_argument('--gpu', type=bool, default=True)

parser.add_argument('--nu', type=float, default=0.05)

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

    with open(default_output_dir() + "/results/" + args.experiment_id + "_args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    util.create_artefact_dirs(default_output_dir(), random_seed)
    # these parameter are changed
    pg_agent_config = PolicyGradientAgentConfig(gamma=1, alpha_attacker=args.alpha_attacker,
                                                #alpha_defender=args.alpha_defender,
                                                epsilon=1, render=False, eval_epsilon=0.,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=args.eval_episodes, train_log_frequency=args.train_log_frequency,
                                                epsilon_decay=0., video=False, eval_log_frequency=args.eval_log_frequency,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos/" + args.experiment_id,
                                                num_episodes=args.num_episodes,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs/" + args.experiment_id,
                                                eval_frequency=args.eval_frequency,
                                                attacker=args.attacker,
                                                #defender=args.defender,
                                                video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data/" + args.experiment_id,
                                                checkpoint_freq=5000,
                                                input_dim_attacker=args.input_dim_attacker,
                                                output_dim_attacker=args.output_dim_attacker,
                                                #input_dim_defender=args.input_dim_defender,
                                                #output_dim_defender=args.output_dim_defender,
                                                hidden_dim=64,
                                                num_hidden_layers=1, batch_size=args.batchsize,
                                                gpu=args.gpu, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard/" + args.experiment_id,
                                                optimizer="Adam", lr_exp_decay=True, lr_decay_rate=args.lr_decay_rate
                                                )
    env_name = args.env_name
    env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + args.experiment_id)
    attacker_agent = BayesReinforceAgent(M=args.M, risk_averse=args.risk_averse, nu=args.nu, static=args.static, env=env, config=pg_agent_config)
    attacker_agent.train()
    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result
