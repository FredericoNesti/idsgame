import os
import gym
import sys
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.bayesian_methods.policy_gradient.reinforce.bayes_reinforce_heuristic import BayesReinforceAgent
from util import util

import argparse
import json

parser = argparse.ArgumentParser(description='Dataset')

# set these parameters in shell file
parser.add_argument('--experiment_id', type=str, default="BayesHeuristic.0.MinDef19.1000.00.123456789")
parser.add_argument('--seed', type=int, default=123456789)
parser.add_argument('--prior_type', type=int, default=3)
parser.add_argument('--gpu', type=bool, default=False)

# bayesian implementation params
parser.add_argument('--prior_state', type=bool, default=True)

# Bayesian Quadrature parametrization
parser.add_argument('--GP_type', type=str, default="scalar_valued")
parser.add_argument('--measure_noise_var', type=float, default=100.)


#kernel parameters
parser.add_argument('--kernel_type', type=str, default="Cauchy")
parser.add_argument('--kernel_var', type=float, default=100.)

#sparsification parameters
parser.add_argument('--M', type=int, default=50)
parser.add_argument('--nu', type=int, default=0.5)
parser.add_argument('--nu_min', type=int, default=0.01)


#exploration parameters
parser.add_argument('--entropy_reg', type=float, default=0.0001)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--min_epsilon', type=float, default=0.01)
parser.add_argument('--epsilon_decay', type=float, default=0.1)

#warm-ups
parser.add_argument('--heuristic_warmup', type=int, default=1000)





#set these for faster implementation
parser.add_argument('--num_episodes', type=int, default=2001)
parser.add_argument('--eval_frequency', type=int, default=100)
parser.add_argument('--train_log_frequency', type=int, default=50)
parser.add_argument('--eval_log_frequency', type=int, default=50)


parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--lr_decay_rate', type=float, default=0.9)
parser.add_argument('--lr_exp_decay', type=bool, default=True)
parser.add_argument('--defender', type=bool, default=False)
parser.add_argument('--attacker', type=bool, default=True)



parser.add_argument('--env_name', type=str, default="idsgame-minimal_defense-v19")
parser.add_argument('--input_dim_attacker', type=int, default=40)
parser.add_argument('--output_dim_attacker', type=int, default=44)
#parser.add_argument('--input_dim_defender', type=int, default=44)
#parser.add_argument('--output_dim_defender', type=int, default=44)
parser.add_argument('--alpha_attacker', type=float, default=0.0001)
parser.add_argument('--alpha_defender', type=float, default=0.0001)
parser.add_argument('--eval_episodes', type=int, default=100)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--discount_factor', type=int, default=0.999)

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
    random_seed = args.seed

    with open(default_output_dir() + "/results/" + args.experiment_id + "_args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    util.create_artefact_dirs(default_output_dir(), args.experiment_id)
    # these parameter are changed
    '''
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
    '''

    pg_agent_config = PolicyGradientAgentConfig(gamma=args.discount_factor, alpha_attacker=args.alpha_attacker, epsilon=args.epsilon, render=False,
                                            alpha_defender=args.alpha_defender,
                                            eval_sleep=0.9,
                                            min_epsilon=args.min_epsilon, eval_episodes=args.eval_episodes, train_log_frequency=args.train_log_frequency,
                                            epsilon_decay=args.epsilon_decay, video=False, eval_log_frequency=args.eval_log_frequency,
                                            video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                            num_episodes=args.num_episodes,
                                            eval_render=False, gifs=False,
                                            gif_dir=default_output_dir() + "/results/gifs/" + args.experiment_id,
                                            eval_frequency=args.eval_frequency, attacker=args.attacker, defender=args.defender,
                                            video_frequency=1001,
                                            save_dir=default_output_dir() + "/results/data/" + args.experiment_id,
                                            checkpoint_freq=500,
                                            input_dim_attacker=((4 + 2) * 4),
                                            output_dim_attacker=(4 + 1) * 4,
                                            input_dim_defender=((4 + 1) * 4),
                                            output_dim_defender=5 * 4,
                                            hidden_dim=args.hidden_dim, num_hidden_layers=args.num_hidden_layers,
                                            pi_hidden_layers=1, pi_hidden_dim=128, vf_hidden_layers=1,
                                            vf_hidden_dim=128,
                                            batch_size=args.batchsize,
                                            gpu=args.gpu, tensorboard=True,
                                            tensorboard_dir=default_output_dir() + "/results/tensorboard/" + args.experiment_id,
                                            optimizer="Adam", lr_exp_decay=args.lr_exp_decay, lr_decay_rate=args.lr_decay_rate,
                                            state_length=1, normalize_features=False, merged_ad_features=True,
                                            zero_mean_features=False,
                                            gpu_id=0,
                                            lstm_network=False,
                                            lstm_seq_length=4, num_lstm_layers=2, optimization_iterations=10,
                                            eps_clip=0.2, max_gradient_norm=0.5, gae_lambda=0.95,
                                            cnn_feature_extractor=False, features_dim=512,
                                            flatten_feature_planes=False, cnn_type=5, vf_coef=0.5, ent_coef=0.001,
                                            render_attacker_view=False, lr_progress_power_decay=4,
                                            lr_progress_decay=False, use_sde=False, sde_sample_freq=4,
                                            one_hot_obs=False, lstm_core=False, lstm_hidden_dim=32,
                                            multi_channel_obs=False,
                                            channel_1_dim=32, channel_1_layers=2, channel_1_input_dim=16,
                                            channel_2_dim=32, channel_2_layers=2, channel_2_input_dim=16,
                                            channel_3_dim=32, channel_3_layers=2, channel_3_input_dim=4,
                                            channel_4_dim=32, channel_4_layers=2, channel_4_input_dim=4,
                                            mini_batch_size=64, ar_policy=False,
                                            attacker_node_input_dim=((4 + 2) * 4),
                                            attacker_at_net_input_dim=(4 + 2), attacker_at_net_output_dim=(4 + 1),
                                            attacker_node_net_output_dim=4)

    env_name = args.env_name
    env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + args.experiment_id)
    attacker_agent = BayesReinforceAgent(prior_state=args.prior_state,
                                         prior_type=args.prior_type,
                                         M=args.M,
                                         nu=args.nu,
                                         nu_min=args.nu_min,
                                         GP_type=args.GP_type,
                                         kernel_type=args.kernel_type,
                                         kernel_var=args.kernel_var,
                                         heuristic_warmup=args.heuristic_warmup,
                                         measure_noise_var=args.measure_noise_var,
                                         env=env,
                                         entropy_reg=args.entropy_reg,
                                         config=pg_agent_config)
    attacker_agent.train()
    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result
