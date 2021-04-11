"""
An agent for the IDSGameEnv that implements the REINFORCE Policy Gradient algorithm.
"""
from typing import Union, List
import numpy as np
import time
import tqdm
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.envs.constants import constants
from gym_idsgame.agents.training_agents.models.fnn_w_softmax import FNNwithSoftmax
from gym_idsgame.agents.training_agents.models.lstm_w_softmax import LSTMwithSoftmax
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent import PolicyGradientAgent
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
#from gym_idsgame.agents.gp_methods import GPTD_step
from numpy.linalg import inv
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as sinv
import numpy as np

class BayesReinforceAgent(PolicyGradientAgent):
    """
    An implementation of the REINFORCE Policy Gradient algorithm
    """
    def __init__(self, env:IdsGameEnv, config: PolicyGradientAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(BayesReinforceAgent, self).__init__(env, config)
        self.attacker_policy_network = None
        self.defender_policy_network = None
        self.loss_fn = None
        self.attacker_optimizer = None
        self.defender_optimizer = None
        self.attacker_lr_decay = None
        self.defender_lr_decay = None
        self.tensorboard_writer = SummaryWriter(self.config.tensorboard_dir)
        self.initialize_models()
        self.tensorboard_writer.add_hparams(self.config.hparams_dict(), {})
        self.machine_eps = np.finfo(np.float32).eps.item()
        self.env.idsgame_config.save_trajectories = False
        self.env.idsgame_config.save_attack_stats = False
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            self.device = device

    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """

        # Initialize models
        if self.config.lstm_network:
            self.attacker_policy_network = LSTMwithSoftmax(input_dim=self.config.input_dim_attacker,
                                                           output_dim=self.config.output_dim_attacker,
                                                           hidden_dim=self.config.hidden_dim,
                                                           num_lstm_layers=self.config.num_lstm_layers,
                                                           num_hidden_linear_layers=self.config.num_hidden_layers,
                                                           hidden_activation="ReLU",
                                                           seq_length=self.config.lstm_seq_length)
            self.defender_policy_network = LSTMwithSoftmax(input_dim=self.config.input_dim_defender,
                                                           output_dim=self.config.output_dim_defender,
                                                           hidden_dim=self.config.hidden_dim,
                                                           num_lstm_layers=self.config.num_lstm_layers,
                                                           num_hidden_linear_layers=self.config.num_hidden_layers,
                                                           hidden_activation="ReLU",
                                                           seq_length=self.config.lstm_seq_length)
        else:
            self.attacker_policy_network = FNNwithSoftmax(input_dim=self.config.input_dim_attacker,
                                                          output_dim=self.config.output_dim_attacker,
                                                          hidden_dim=self.config.hidden_dim,
                                                          num_hidden_layers=self.config.num_hidden_layers,
                                                          hidden_activation=self.config.hidden_activation)
            self.defender_policy_network = FNNwithSoftmax(input_dim=self.config.input_dim_defender,
                                                          output_dim=self.config.output_dim_defender,
                                                          hidden_dim=self.config.hidden_dim,
                                                          num_hidden_layers=self.config.num_hidden_layers,
                                                          hidden_activation=self.config.hidden_activation)

        # Specify device
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            self.config.logger.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            self.config.logger.info("Running on the CPU")

        self.attacker_policy_network.to(device)
        self.defender_policy_network.to(device)

        # Define Optimizer. The call to model.parameters() in the optimizer constructor will contain the learnable
        # parameters of the layers in the model
        if self.config.optimizer == "Adam":
            self.attacker_optimizer = torch.optim.Adam(self.attacker_policy_network.parameters(), lr=self.config.alpha_attacker)
            self.defender_optimizer = torch.optim.Adam(self.defender_policy_network.parameters(), lr=self.config.alpha_defender)
        elif self.config.optimizer == "SGD":
            self.attacker_optimizer = torch.optim.SGD(self.attacker_policy_network.parameters(), lr=self.config.alpha_attacker)
            self.defender_optimizer = torch.optim.SGD(self.defender_policy_network.parameters(), lr=self.config.alpha_defender)
        else:
            raise ValueError("Optimizer not recognized")

        # LR decay
        if self.config.lr_exp_decay:
            self.attacker_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.attacker_optimizer,
                                                                       gamma=self.config.lr_decay_rate)
            self.defender_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.attacker_optimizer,
                                                                            gamma=self.config.lr_decay_rate)

    def get_action(self, state: np.ndarray, attacker : bool = True, legal_actions : List = None,
                   non_legal_actions : List = None) -> Union[int, torch.Tensor]:
        """
        Samples an action from the policy network

        :param state: the state to sample an action for
        :param attacker: boolean flag whether running in attacker mode (if false assume defender)
        :param legal_actions: list of allowed actions
        :param non_legal_actions: list of disallowed actions
        :return: The sampled action id
        """

        print('YES')

        if self.config.lstm_network:
            state = torch.from_numpy(state.reshape(1, state.shape[0], state.shape[1]*state.shape[2])).float()
        else:
            state = torch.from_numpy(state.flatten()).float()

        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            state = state.to(device)

        # Calculate legal actions
        if attacker:
            actions = list(range(self.env.num_attack_actions))
            if not self.env.local_view_features() or (legal_actions is None or non_legal_actions is None):
                legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
                non_legal_actions = list(filter(lambda action: not self.env.is_attack_legal(action), actions))
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))
            non_legal_actions = list(filter(lambda action: not self.env.is_defense_legal(action), actions))

        # Forward pass using the current policy network to predict P(a|s)
        if attacker:
            action_probs = self.attacker_policy_network(state).squeeze()
            # Set probability of non-legal actions to 0
            action_probs_1 = action_probs.clone()
            if len(legal_actions) > 0 and len(non_legal_actions) < self.env.num_attack_actions:
                action_probs_1[non_legal_actions] = 0
        else:
            action_probs = self.defender_policy_network(state).squeeze()
            # Set probability of non-legal actions to 0
            action_probs_1 = action_probs.clone()
            # print("state shape:{}".format(state.shape))
            # print("action shape:{}".format(action_probs_1.shape))
            if len(legal_actions) > 0 and len(non_legal_actions) < self.env.num_defense_actions:
                action_probs_1[non_legal_actions] = 0

        # Use torch.distributions package to create a parameterizable probability distribution of the learned policy
        # PG uses a trick to turn the gradient into a stochastic gradient which we can sample from in order to
        # approximate the true gradient (which we can’t compute directly). It can be seen as an alternative to the
        # reparameterization trick
        policy_dist = Categorical(action_probs_1)

        # Sample an action from the probability distribution
        try:
            print('policy_dist', policy_dist)
            action = policy_dist.sample()
            print(action)
        except Exception as e:
            print("Nan values in distribution, consider using a lower learnign rate or gradient clipping")
            print("legal actions: {}".format(legal_actions))
            print("non_legal actions: {}".format(non_legal_actions))
            print("action_probs: {}".format(action_probs))
            print("action_probs_1: {}".format(action_probs_1))
            print("state: {}".format(state))
            print("policy_dist: {}".format(policy_dist))
            action = torch.tensor(0).type(torch.LongTensor)

        # log_prob returns the log of the probability density/mass function evaluated at value.
        # save the log_prob as it will use later on for computing the policy gradient
        # policy gradient theorem says that the stochastic gradient of the expected return of the current policy is
        # the log gradient of the policy times the expected return, therefore we save the log of the policy distribution
        # now and use it later to compute the gradient once the episode has finished.
        print('')
        print(policy_dist)
        print(action)

        log_prob = policy_dist.log_prob(action)
        print(log_prob)
        print(action.item())
        return action.item(), log_prob


    def log_prob_func_attacker(self, state: np.ndarray) -> torch.Tensor:
        """
        This function is exclusively useful for Fisher INfo matrix implementation

        :param state: the state to sample an action for
        :param attacker: boolean flag whether running in attacker mode (if false assume defender)
        :return: log_prob
        """
        state = torch.flatten(state, start_dim=1)

        log_prob = torch.zeros(size = (state.shape[0], 1), requires_grad=True)
        if torch.cuda.is_available() and self.config.gpu:
            device   = torch.device("cuda:" + str(self.config.gpu_id))
            log_prob = torch.Tensor(log_prob).to(device)

        for i in range(state.shape[0]):
            action_probs = self.attacker_policy_network(state[i,:]).squeeze()
            policy_dist = Categorical(action_probs)

            # Sample an action from the probability distribution
            action = policy_dist.sample()

            # log_prob returns the log of the probability density/mass function evaluated at value.
            # save the log_prob as it will use later on for computing the policy gradient
            # policy gradient theorem says that the stochastic gradient of the expected return of the current policy is
            # the log gradient of the policy times the expected return, therefore we save the log of the policy distribution
            # now and use it later to compute the gradient once the episode has finished.
            log_prob[i, 0] = policy_dist.log_prob(action)

        return log_prob

    def BPG_eval(self, sample_size, sigma=10, model_type=1):
        """
        This is a Gaussian Process step to be done recursively whenever we want to compute an integral with the help of Bayes Quadrature method.
        The correspondent integral might be related to loss function to be computed and backpropagated or a gradient distribution
        """
        policy_loss = []
        num_batches = self.batch_size
        attacker = True

        for batch in range(num_batches):
            logP = []
            A = []
            X = []
            R = []
            Kinvprev = None
            #Ginv = self.Fisher_inv()
            flag0 = True
            for m in range(sample_size):
                dlogp = []
                a = []
                x = []

                # Sample a path z_i  using the policy \mi(\theta)
                done = False
                obs = self.env.reset(update_stats=False)
                attacker_obs, defender_obs = obs
                attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[],
                                                   attacker=True)
                defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=[],
                                                   attacker=False)
                episode_attacker_reward = 0
                episode_defender_reward = 0
                saved_attacker_log_probs = []
                saved_defender_log_probs = []

                while not done:
                    if self.config.render:
                        self.env.render(mode="human")

                    # Default initialization
                    attacker_action = 0
                    defender_action = 0
                    if self.config.attacker:
                        attacker_action, attacker_log_prob = self.get_action(attacker_state, attacker=True)
                        saved_attacker_log_probs.append(attacker_log_prob)

                        dlogp.append(attacker_log_prob)
                        a.append(attacker_action)
                        x.append(attacker_state)

                    if self.config.defender:
                        defender_action, defender_log_prob = self.get_action(defender_state, attacker=False)
                        saved_defender_log_probs.append(defender_log_prob)

                        dlogp.append(defender_log_prob)
                        a.append(defender_action)
                        x.append(defender_state)

                    action = (attacker_action, defender_action)
                    obs_prime, reward, done, _ = self.env.step(action)

                    # Update state information and metrics
                    attacker_reward, defender_reward = reward
                    obs_prime_attacker, obs_prime_defender = obs_prime
                    episode_attacker_reward += attacker_reward
                    episode_defender_reward += defender_reward
                    attacker_obs = obs_prime_attacker
                    defender_obs = obs_prime_defender
                    attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs,
                                                       state=attacker_state, attacker=True)
                    defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs,
                                                       state=defender_state, attacker=False)

                ### Dictionary Update
                if flag0:
                    flag0 = False
                    # R
                    if self.config.attacker: R.append(episode_attacker_reward)
                    if self.config.defender: R.append(episode_defender_reward)

                    logP.append(torch.Tensor(dlogp).to('cuda'))
                    A.append(torch.Tensor(a).to('cuda'))
                    X.append(torch.Tensor(x).to('cuda'))

                Cinv, Kinv, flag = self.Kernel_Online(path_dict=X, Ginv=Ginv, Kprev=Kinvprev, episode=m,
                                                      max_episode=sample_size - 1)
                Kinvprev = Kinv

                ### Dictionary Update
                if flag:
                    flag0 = False
                    # R
                    if self.config.attacker: R.append(episode_attacker_reward)
                    if self.config.defender: R.append(episode_defender_reward)

                    logP.append(torch.Tensor(dlogp).to('cuda'))
                    A.append(torch.Tensor(a).to('cuda'))
                    X.append(torch.Tensor(x).to('cuda'))

            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                R = torch.Tensor(R).to(device)

            FisherINfo = self.Fisher_Info_vs2_check(State=X)
            Ginv = torch.inverse(FisherINfo)

            Y = self.YM(X, logP, R, tgt_size=Ginv.shape[0], model_type=model_type)
            Z = self.ZM(X, logP, Ginv, model_type=model_type)
            Post_Covar = 0
            # z0 = 1 + sample_size
            Post_mean = (Y @ Cinv @ Z) * (model_type == 1)  # + (Z @ Cinv @ Y) * (model_type == 2)
            # Post_Covar = (z0 - Z.T @ C @ Z) @ np.eye(Post_mean.size)
            # if model_type == 2:
            # z0 = Fisher_Info
            # Post_mean  = Z @ C @ Y
            # Post_Covar = z0 - Z @ C @ Z.T

            # loss = - 0.5 * (Post_mean.T @ Post_mean)
            loss = - Post_mean * R / 1
            policy_loss.append(loss)

        # Compute gradient and update models
        if attacker:
            # reset gradients
            self.attacker_optimizer.zero_grad()
            # expected loss over the batch
            policy_loss_total = torch.stack(policy_loss).sum()
            policy_loss_final = policy_loss_total / num_batches
            policy_loss_final.retain_grad()
            policy_loss_final.backward()
            # maybe clip gradient
            if self.config.clip_gradient:
                torch.nn.utils.clip_grad_norm_(self.attacker_policy_network.parameters(), 1)
            # gradient descent step
            self.attacker_optimizer.step()
        return policy_loss_final

    #####################################################################################




    def BPG_step(self, X, logP, R, Ginv, model_type, Cinv):
        Y = self.YM(X, logP, R, tgt_size=Ginv.shape[0], model_type=model_type)
        Z = self.ZM(X, logP, Ginv, model_type=model_type)
        #print(Y.shape)
        #print(Cinv.shape)
        Post_mean = (Y @ Cinv @ Z) * (model_type == 1)  # + (Z @ Cinv @ Y) * (model_type == 2)
        # loss = - 0.5 * (Post_mean.T @ Post_mean)
        loss = - Post_mean * R / 1
        #policy_loss.append(loss)
        return loss



    #def training_step(self, saved_rewards : List[List[float]], saved_log_probs : List[List[torch.Tensor]],
    #                  attacker=True) -> torch.Tensor:
    def training_step(self, X, logP, R, Ginv, Cinv, model_type=1,attacker=True):
        """
        Performs a training step of the Deep-Q-learning algorithm (implemented in PyTorch)

        :param saved_rewards list of rewards encountered in the latest episode trajectory
        :param saved_log_probs list of log-action probabilities (log p(a|s)) encountered in the latest episode trajectory
        :return: loss
        """
        #print("saved rewards: {}".format(sum(sum(x) for x in saved_rewards)))
        policy_loss = []

        #num_batches = len(saved_rewards)
        num_batches = self.config.batch_size

        for batch in range(num_batches):
            '''
            R = 0
            returns = []

            # Create discounted returns. When episode is finished we can go back and compute the observed cumulative
            # discounted reward by using the observed rewards
            for r in saved_rewards[batch][::-1]:
                R = r + self.config.gamma * R
                returns.insert(0, R)
            num_rewards = len(returns)

            # convert list to torch tensor
            returns = torch.tensor(returns)

            # normalize
            std = returns.std()
            if num_rewards < 2:
                std = 0
            returns = (returns - returns.mean()) / (std + self.machine_eps)

            # Compute PG "loss" which in reality is the expected reward, which we want to maximize with gradient ascent
            for log_prob, R in zip(saved_log_probs[batch], returns):
                # negative log prob since we are doing gradient descent (not ascent)
                policy_loss.append(-log_prob * R)
            '''
            policy_loss0 = self.BPG_step(X, logP, R, Ginv, model_type, Cinv)
            policy_loss.append(policy_loss0)

        # Compute gradient and update models
        if attacker:
            # reset gradients
            self.attacker_optimizer.zero_grad()
            # expected loss over the batch
            policy_loss_total = torch.stack(policy_loss).sum()
            policy_loss = policy_loss_total/num_batches
            # perform backprop
            policy_loss.backward()
            # maybe clip gradient
            if self.config.clip_gradient:
                torch.nn.utils.clip_grad_norm_(self.attacker_policy_network.parameters(), 1)
            # gradient descent step
            self.attacker_optimizer.step()

        else:
            # reset gradients
            self.defender_optimizer.zero_grad()
            # expected loss over the batch
            policy_loss_total = torch.stack(policy_loss).sum()
            policy_loss = policy_loss_total/num_batches
            # perform backprop
            policy_loss.backward()
            # maybe clip gradient
            if self.config.clip_gradient:
                torch.nn.utils.clip_grad_norm_(self.defender_policy_network.parameters(), 1)
            # gradient descent step
            self.defender_optimizer.step()

        return policy_loss


    def train(self) -> ExperimentResult:
        """
        Runs the REINFORCE algorithm

        :return: Experiment result
        """
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs
        attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[],
                                           attacker=True)
        defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=[],
                                           attacker=False)

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []
        episode_avg_attacker_loss = []
        episode_avg_defender_loss = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        saved_attacker_log_probs_batch = []
        saved_attacker_rewards_batch = []
        saved_defender_log_probs_batch = []
        saved_defender_rewards_batch = []


        # Training
        #print('NUm epidsodes', self.config.num_episodes)
        for iter in range(self.config.num_episodes):
            #print('episode', iter)
            ### For Dictionary Update
            flag0 = True
            M = 30
            Kinvprev = None
            Ginv = self.Fisher_inv()
            logP = []
            A = []
            X = []
            R = []
            # Batch
            #for episode in range(self.config.batch_size):
            for m in range(M):
                dlogp = []
                a = []
                x = []

                episode_attacker_reward = 0
                episode_defender_reward = 0
                episode_step = 0
                episode_attacker_loss = 0.0
                episode_defender_loss = 0.0
                saved_attacker_log_probs = []
                saved_attacker_rewards = []
                saved_defender_log_probs = []
                saved_defender_rewards = []
                while not done:
                    if self.config.render:
                        self.env.render(mode="human")

                    if not self.config.attacker and not self.config.defender:
                        raise AssertionError("Must specify whether training an attacker agent or defender agent")

                    # Default initialization
                    attacker_action = 0
                    defender_action = 0

                    # Get attacker and defender actions
                    if self.config.attacker:
                        legal_actions = None
                        illegal_actions = None
                        if self.env.local_view_features():
                            legal_actions, illegal_actions = self.get_legal_attacker_actions(attacker_obs)
                        attacker_action, attacker_log_prob = self.get_action(attacker_state, attacker=True,
                                                                             legal_actions=legal_actions,
                                                                             non_legal_actions=illegal_actions)
                        if self.env.local_view_features():
                            attacker_action = PolicyGradientAgent.convert_local_attacker_action_to_global(attacker_action, attacker_obs)
                        saved_attacker_log_probs.append(attacker_log_prob)

                        dlogp.append(attacker_log_prob)
                        a.append(attacker_action)
                        x.append(attacker_state)

                    print('here')
                    if self.config.defender:
                        print(defender_state)
                        defender_action, defender_log_prob= self.get_action(defender_state, attacker=False)
                        saved_defender_log_probs.append(defender_log_prob)

                        dlogp.append(defender_log_prob)
                        a.append(defender_action)
                        x.append(defender_state)

                    action = (attacker_action, defender_action)

                    # Take a step in the environment
                    obs_prime, reward, done, _ = self.env.step(action)

                    # Update metrics
                    attacker_reward, defender_reward = reward
                    obs_prime_attacker, obs_prime_defender = obs_prime
                    episode_attacker_reward += attacker_reward
                    saved_attacker_rewards.append(attacker_reward)
                    episode_defender_reward += defender_reward
                    saved_defender_rewards.append(defender_reward)
                    episode_step += 1

                    # Move to the next state
                    obs = obs_prime
                    attacker_obs = obs_prime_attacker
                    defender_obs = obs_prime_defender
                    attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs,
                                                       state=attacker_state, attacker=True)
                    defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs,
                                                       state=defender_state, attacker=False)

                # Render final frame
                if self.config.render:
                    self.env.render(mode="human")

                ### First Dictionary Update
                if flag0:
                    flag0 = False

                    # R
                    if self.config.attacker: R.append(episode_attacker_reward)
                    if self.config.defender: R.append(episode_defender_reward)

                    logP.append(torch.Tensor(dlogp).to('cuda'))
                    A.append(torch.Tensor(a).to('cuda'))
                    X.append(torch.Tensor(x).to('cuda'))

                    # Accumulate batch
                    saved_attacker_log_probs_batch.append(saved_attacker_log_probs)
                    saved_attacker_rewards_batch.append(saved_attacker_rewards)
                    saved_defender_log_probs_batch.append(saved_defender_log_probs)
                    saved_defender_rewards_batch.append(saved_defender_rewards)
                ### End First Dictionary Update

                Cinv, Kinv, flag = self.Kernel_Online(path_dict=X, Ginv=Ginv, Kprev=Kinvprev, episode=m,
                                                      max_episode=M - 1)
                Kinvprev = Kinv

                ### Dictionary Update
                if flag:
                    flag0 = False
                    # R
                    if self.config.attacker: R.append(episode_attacker_reward)
                    if self.config.defender: R.append(episode_defender_reward)

                    logP.append(torch.Tensor(dlogp).to('cuda'))
                    A.append(torch.Tensor(a).to('cuda'))
                    X.append(torch.Tensor(x).to('cuda'))

                    # Accumulate batch
                    saved_attacker_log_probs_batch.append(saved_attacker_log_probs)
                    saved_attacker_rewards_batch.append(saved_attacker_rewards)
                    saved_defender_log_probs_batch.append(saved_defender_log_probs)
                    saved_defender_rewards_batch.append(saved_defender_rewards)
                ### End Dictionary Update


                # Record episode metrics
                self.num_train_games += 1
                self.num_train_games_total += 1
                if self.env.state.hacked:
                    self.num_train_hacks += 1
                    self.num_train_hacks_total += 1

                episode_attacker_rewards.append(episode_attacker_reward)
                episode_defender_rewards.append(episode_defender_reward)
                episode_steps.append(episode_step)

                # Reset environment for the next episode and update game stats
                done = False
                attacker_obs, defender_obs = self.env.reset(update_stats=True)
                attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[],
                                                   attacker=True)
                defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=[],
                                                   attacker=False)

            # Once done saving rewards send it to cuda
            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                R = torch.Tensor(R).to(device)


            # Perform Batch Policy Gradient updates
            if self.config.attacker:
                #loss = self.training_step(saved_attacker_rewards_batch, saved_attacker_log_probs_batch, attacker=True)
                loss = self.training_step(X, logP, R, Ginv, Cinv=Cinv, attacker=True)
                episode_attacker_loss += loss.item()

            if self.config.defender:
                #loss = self.training_step(saved_defender_rewards_batch, saved_defender_log_probs_batch, attacker=False)
                loss = self.training_step(X, logP, R, Ginv, Cinv=Cinv, attacker=False)
                episode_defender_loss += loss.item()

            if self.config.batch_size > 0:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss / self.config.batch_size)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss / self.config.batch_size)
            else:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss)

            # Reset batch
            saved_attacker_log_probs_batch = []
            saved_attacker_rewards_batch = []
            saved_defender_log_probs_batch = []
            saved_defender_rewards_batch = []

            # Decay LR after every iteration
            lr_attacker = self.config.alpha_attacker
            if self.config.lr_exp_decay:
                self.attacker_lr_decay.step()
                lr_attacker = self.attacker_lr_decay.get_lr()[0]

            # Decay LR after every iteration
            lr_defender = self.config.alpha_defender
            if self.config.lr_exp_decay:
                self.defender_lr_decay.step()
                lr_defender = self.defender_lr_decay.get_lr()[0]


            # Log average metrics every <self.config.train_log_frequency> iterations
            if iter % self.config.train_log_frequency == 0:
                if self.num_train_games > 0 and self.num_train_games_total > 0:
                    self.train_hack_probability = self.num_train_hacks / self.num_train_games
                    self.train_cumulative_hack_probability = self.num_train_hacks_total / self.num_train_games_total
                else:
                    self.train_hack_probability = 0.0
                    self.train_cumulative_hack_probability = 0.0
                self.log_metrics(iteration=iter, result=self.train_result, attacker_episode_rewards=episode_attacker_rewards,
                                 defender_episode_rewards=episode_defender_rewards, episode_steps=episode_steps,
                                 episode_avg_attacker_loss=episode_avg_attacker_loss, episode_avg_defender_loss=episode_avg_defender_loss,
                                 eval=False, update_stats=True, lr_attacker=lr_attacker, lr_defender=lr_defender)

                # Log values and gradients of the parameters (histogram summary) to tensorboard
                if self.config.attacker:
                    for tag, value in self.attacker_policy_network.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), iter)
                        self.tensorboard_writer.add_histogram(tag + '_attacker/grad', value.grad.data.cpu().numpy(),
                                                              iter)

                if self.config.defender:
                    for tag, value in self.defender_policy_network.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), iter)
                        self.tensorboard_writer.add_histogram(tag + '_defender/grad', value.grad.data.cpu().numpy(),
                                                              iter)

                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []
                self.num_train_games = 0
                self.num_train_hacks = 0

            # Run evaluation every <self.config.eval_frequency> iterations
            if iter % self.config.eval_frequency == 0:
                self.eval(iter)

            # Save models every <self.config.checkpoint_frequency> iterations
            if iter % self.config.checkpoint_freq == 0:
                self.save_model()
                self.env.save_trajectories(checkpoint=True)
                self.env.save_attack_data(checkpoint=True)
                if self.config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

            self.outer_train.update(1)

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        self.eval(self.config.num_episodes-1, log=False)

        # Save networks
        self.save_model()

        # Save other game data
        self.env.save_trajectories(checkpoint = False)
        self.env.save_attack_data(checkpoint=False)
        if self.config.save_dir is not None:
            time_str = str(time.time())
            self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        return self.train_result


    def eval(self, train_episode, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param train_episode: the train episode to keep track of logging
        :param log: whether to log the result
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

        self.num_eval_games = 0
        self.num_eval_hacks = 0

        if len(self.eval_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting eval with non-empty result object")
        if self.config.eval_episodes < 1:
            return
        done = False

        # Video configif self.config.attacker: R = episode_attacker_reward
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='Eval Episode', position=1)
        self.outer_eval.set_description_str(
            "[Eval] avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
            "acc_D_R:{:.2f}".format(0.0, 0,0, 0.0, 0.0, 0.0, 0.0))

        # Eval
        attacker_obs, defender_obs = self.env.reset(update_stats=False)
        attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[],
                                           attacker=True)
        defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=[],
                                           attacker=False)

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    legal_actions = None
                    illegal_actions = None
                    if self.env.local_view_features():
                        legal_actions, illegal_actions = self.get_legal_attacker_actions(attacker_obs)
                    attacker_action, _ = self.get_action(attacker_state, attacker=True,
                                                         legal_actions=legal_actions, non_legal_actions=illegal_actions)
                    if self.env.local_view_features():
                        attacker_action = PolicyGradientAgent.convert_local_attacker_action_to_global(attacker_action, attacker_obs)
                if self.config.defender:
                    defender_action, _ = self.get_action(defender_state, attacker=False)
                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender
                attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs,
                                                   state=attacker_state, attacker=True)
                defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs,
                                                   state=defender_state, attacker=False)

            # Render final frame when game completed
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, episode_step))

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update eval stats
            self.num_eval_games += 1
            self.num_eval_games_total += 1
            if self.env.state.detected:
                self.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            if self.env.state.hacked:
                self.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.num_eval_hacks += 1
                self.num_eval_hacks_total += 1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                if self.num_eval_games_total > 0:
                    self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                        self.num_eval_games_total)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 eval = True, update_stats=False)

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(train_episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

                # Add frames to tensorboard
                for idx, frame in enumerate(self.env.episode_frames):
                    self.tensorboard_writer.add_image(str(train_episode) + "_eval_frames/" + str(idx),
                                                       frame, global_step=train_episode,
                                                      dataformats = "HWC")


            # Reset for new eval episode
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=False)
            attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs,
                                               state=attacker_state, attacker=True)
            defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs,
                                               state=defender_state, attacker=False)
            self.outer_eval.update(1)

        # Log average eval statistics
        if log:
            if self.num_eval_hacks > 0:
                self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
            if self.num_eval_games_total > 0:
                self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                    self.num_eval_games_total)

            self.log_metrics(train_episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, eval=True, update_stats=True)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def save_model(self) -> None:
        """
        Saves the PyTorch Model Weights

        :return: None
        """
        time_str = str(time.time())
        if self.config.save_dir is not None:
            if self.config.attacker:
                path = self.config.save_dir + "/" + time_str + "_attacker_policy_network.pt"
                self.config.logger.info("Saving policy-network to: {}".format(path))
                torch.save(self.attacker_policy_network.state_dict(), path)
            if self.config.defender:
                path = self.config.save_dir + "/" + time_str + "_defender_policy_network.pt"
                self.config.logger.info("Saving policy-network to: {}".format(path))
                torch.save(self.defender_policy_network.state_dict(), path)
        else:
            self.config.logger.warning("Save path not defined, not saving policy-networks to disk")

    def score_func(self, state):
        #grad_out = jacobian(func=self.log_prob_func_attacker, inputs=state)#.reshape(-1,1)
        #state = Variable(state, requires_grad=True)
        model = self.attacker_policy_network
        y = self.log_prob_func_attacker(state)
        #print('SCORE FUNC')
        #print(state.requires_grad)
        #print(y.requires_grad)
        #grad_out = torch.Tensor(list(map(lambda y, param: gradient(y, param), y, model.parameters())))
        #print(len(model.parameters()))
        grad_out = torch.Tensor([])
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            grad_out = grad_out.to(device)

        for param in model.parameters():
            #print(gradient(y, param).flatten())
            grad_out = torch.hstack((grad_out, gradient(y, param).flatten()))
        #print(grad_out)
        #grad_out = torch.stack(grad_out, dim=0)
        #grad_out = gradient(y, model.parameters())
        #print('Inside score_func')
        #print(grad_out)
        #print(grad_out)
        #grad_out = Variable(grad_out, requires_grad=True)
        return grad_out

    def Fisher_Info_vs2_check(self, State):
        d = 0
        flag = False
        #print('INside FIM check')
        #if torch.cuda.is_available() and self.config.gpu:
        #    device = torch.device("cuda:" + str(self.config.gpu_id))
        #    Fisherinfo = Fisherinfo.to(device)
        for x in State:
            #print(x)
            d += len(x)
            uz = self.score_func(x).reshape(-1,1)
            #print(uz.shape)
            #print('')
            #print('again')
            #print(uz)
            if flag == False:
                #print(uz.shape)
                Fisherinfo = uz @ uz.T
            if flag == True:
                Fisherinfo += uz @ uz.T
                flag = True
        Fisherinfo /= d
        Fisherinfo += torch.eye(n=Fisherinfo.shape[0], m=Fisherinfo.shape[1], device=Fisherinfo.device)
        return Fisherinfo

    def ZM(self, X, A, Ginv, model_type = 1):
        #if model_type == 1:
        zM = Variable(torch.zeros(1, len(X)), requires_grad=True)
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            zM = zM.to(device)
        #if model_type == 2:
        #    zM = torch.zeros(2, len(X))
        #    if torch.cuda.is_available() and self.config.gpu:
        #        device = torch.device("cuda:" + str(self.config.gpu_id))
        #        zM = zM.to(device)
        for i in range(len(X)):
            #uzi = self.score_vector(a=A[i], mx=torch.mean(X[i]), sx=torch.var(X[i]))
            uzi = self.score_func(state=X[i]).reshape(-1,1)
            #if model_type == 1:
            zM[:,i] = 1 + uzi.T @ Ginv @ uzi
            #if model_type == 2:
            #    zM[:,i] = uzi[:,0]
        return zM.T

    def YM(self, X, A, R,tgt_size, model_type=1):
        #if model_type == 1:
        yM = Variable(torch.zeros(tgt_size, len(X)), requires_grad=True)
        #print('y', yM.requires_grad)
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            yM = yM.to(device)
        #if model_type == 2:
        #    yM = torch.zeros(1, len(X))
        #    if torch.cuda.is_available() and self.config.gpu:
        #        device = torch.device("cuda:" + str(self.config.gpu_id))
        #        yM = yM.to(device)

        for i in range(len(X)):
            #uzi = self.score_vector(a=A[i], mx=torch.mean(X[i]), sx=torch.var(X[i]))
            uzi = self.score_func(state=X[i])
            #if model_type == 1:
                #print(R.shape)
                #print(uzi.shape)
            yM[:,i] = R[i] * uzi.reshape(-1,1)[:,0] #### IMPORTANT REWARD MIGHT BE WRONG
            #if model_type == 2:
            #    yM[:,i] = R[i]
        return yM



    def Kernel_Online(self, path_dict, Ginv, Kprev = None, sigma2 = 100, nu = 0.01, model_type = 1, episode = None, max_episode = None):
        if Kprev == None:
            flag_dictionary = False
            C_inv = None
            uz1 = self.score_func(state=path_dict[0])
            k11 = torch.pow(1 + uz1.T @ Ginv @ uz1, 2) * (model_type == 1)
            K = Variable(torch.Tensor([1/k11]), requires_grad=True).reshape(1,1)
            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                K = K.to(device)

            return C_inv, K, flag_dictionary
        elif Kprev != None:
            #print(path_dict)
            #print(len(path_dict))
            k = torch.zeros(len(path_dict),1)
            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                k = k.to(device)
            uzj = self.score_func(state=path_dict[-1])
            for t in range(len(path_dict)-1):
                uzi = self.score_func(state = path_dict[t])
                k[t,0] = torch.pow(1 + uzi.T @ Ginv @ uzj, 2) * (model_type == 1) #+ uzi.T @ Ginv @ uzj * (model_type == 2)

            #print('a')
            #print(Kprev)
            #print(k)
            a = Kprev @ k
            delta = (Kprev[-1,-1] - k.T @ a)
            #print('compare here')
            #print(delta)
            #print(Kprev)
            if delta > nu:
                #print('Yes')
                flag_dictionary = True
                aux = torch.Tensor([1]).to(self.device)
                K = (1/delta) * torch.hstack([torch.vstack([(delta*Kprev + a @ a.T), -a.T]), torch.vstack([-a, aux])])
            else:
                #print('No')
                K = Kprev
                flag_dictionary = False

            if episode == max_episode:
                #print('check Cinv shape', K.shape)
                Sigma = (sigma2)*torch.eye(K.shape[0],K.shape[1])
                if torch.cuda.is_available() and self.config.gpu:
                    device = torch.device("cuda:" + str(self.config.gpu_id))
                    Sigma = Sigma.to(device)

                #C_inv = (K + Sigma_inv) @ Sigma_inv @ K
                C_inv = torch.inverse((1/sigma2)*Sigma + K @ Sigma)
            else:
                C_inv = None

            return C_inv, K, flag_dictionary

    def Fisher_inv(self, sigma=1, n=100):
        #auxw = []
        #for param in self.attacker_policy_network.parameters():
        #    auxw.append(param[0])

        auxw = torch.Tensor([])
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            auxw = auxw.to(device)

        for param in self.attacker_policy_network.parameters():
            #print(gradient(y, param).flatten())
            auxw = torch.hstack((auxw, param.flatten()))

        #print('HERE')
        #print(auxw)
        weight_vector = auxw
        #print(weight_vector)
        dim = weight_vector.shape[0]
        #print('inside fisher inv')
        #print(dim)
        Ginv = torch.zeros(dim, dim)
        I = torch.eye(dim, dim)
        samples_normal = torch.from_numpy(np.random.normal(size=n))

        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            Ginv = Ginv.to(device)
            I = I.to(device)
            samples_normal = samples_normal.to(device)

        norm = torch.linalg.norm(weight_vector)
        norm2 = norm ** 2


        aux = 1 / (4 * np.sqrt(2 * np.pi) * (sigma ** 2) * norm2)

        c1w = 0
        c2w = 0
        for i in range(n):
            c1w += aux * f(norm * samples_normal[i])
            c2w += aux * f(norm * samples_normal[i]) * samples_normal[i] ** 2
        c1w /= n
        c2w /= n

        Ginv += (1 / norm2 * c1w) * I + (1 / norm2 ** 2) * (1 / c2w - 1 / c1w) * weight_vector @ weight_vector.T
        #print('out Ginv')
        return Ginv #+ 0.5*I


def gradient(y, x, grad_outputs=None):
    # Source: https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240/3
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

def f(u):
    return (((1 - torch.exp(-u))/(1 + torch.exp(-u)))**2 - 1)**2









