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
import numpy as np
import tensorflow as tf
#   import tfplot
import numpy as np
import os.path
#import scipy.misc
#import scipy.ndimage
#import skimage.data
#import seaborn.apionly as sns

#seed0 = 670979600
#torch.manual_seed(seed0)

class BayesReinforceAgent(PolicyGradientAgent):
    """
    An implementation of the REINFORCE Policy Gradient algorithm
    """
    def __init__(self, prior_state:bool, prior_type:int, M:int, env:IdsGameEnv, config: PolicyGradientAgentConfig):
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

        self.prior_state = prior_state
        self.prior_type = prior_type
        self.M = M


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
        if self.config.lstm_network:
            state = torch.from_numpy(state.reshape(1, state.shape[0], state.shape[1]*state.shape[2])).float()
        else:
            state = Variable(torch.from_numpy(state.flatten()).float(), requires_grad=True)

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

            #print('')
            #print('state', state)

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

        #action_probs_1 /= torch.sum(action_probs_1)

        #print('action probs 1', action_probs_1)
        #print('sum of probs', torch.sum(action_probs_1))

        policy_dist = Categorical(action_probs_1)

        #print('policy_dist', policy_dist)

        # Sample an action from the probability distribution
        try:
            '''
            if (np.random.random() < self.config.epsilon and not eval) \
                    or (eval and np.random.random() < self.config.eval_epsilon):
                #aux = np.array(np.random.choice(legal_actions))
                action = torch.multinomial(legal_actions, num_samples=1)
                #print('action', action)
                #print(aux)
                #action = Variable(torch.from_numpy(aux), requires_grad=True) #.type(torch.LongTensor)
                if torch.cuda.is_available() and self.config.gpu:
                    device = torch.device("cuda:" + str(self.config.gpu_id))
                    action = action.to(device)
                #action = torch.multinomial(legal_actions, num_samples=1)
                #action2 = policy_dist.sample()
                #print('')
                #print(action)
                #print(action2)
            else:
            '''
            action = policy_dist.sample()

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

        #print('action', action)

        log_prob = policy_dist.log_prob(action)

        #print(log_prob)

        #log_prob.backward(retain_graph=True)
        #print('log_prob', log_prob)
        log_prob.backward()
        i = 0
        for param in self.attacker_policy_network.parameters():

            #print('')
            #print('CHECK OVER GRADS')
            #print(param.grad.data)

            if i == 0:
                i = 1
                if param.grad is not None:
                    grad_out = param.grad.data.flatten()
            else:
                if param.grad is not None:
                    grad_out = torch.hstack((grad_out, param.grad.data.flatten()))

        return action.item(), log_prob, grad_out



    def training_step(self, R, Post_mean, lr, Fisher_inv, log_probs, attacker=True):
        """
        Performs a training step of the Deep-Q-learning algorithm (implemented in PyTorch)

        :param saved_rewards list of rewards encountered in the latest episode trajectory
        :param saved_log_probs list of log-action probabilities (log p(a|s)) encountered in the latest episode trajectory
        :return: loss
        """
        policy_loss = []
        num_batches = self.config.batch_size

        for batch in range(num_batches):

            '''
            R1 = torch.Tensor(R[ti])
            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                R1 = R1.to(device)
            '''

            returns = torch.Tensor([0.]*len(R))

            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                returns = torch.Tensor(returns).to(device)
            R1 = 0
            for i,r in enumerate(R):
                r1 = sum(r)
                R1 = r1 + self.config.gamma * R1
                returns[-(i+1)] += R1


            if len(R) > 1:
                returns = ( returns - returns.mean() )  #/ ( returns.std() + self.machine_eps )


            #policy_loss0 = self.BPG_step(X, logP, returns, Ginv, model_type, Cinv)
            loss = - Post_mean[batch] * returns #- sum(log_probs[batch])/len(log_probs[batch])
            # posso implementar gradiente direto
            policy_loss.append(loss)



        # ALTERNATIVE COMPUTATION OF GRADIENTS

        Post_mean2 = torch.mean(torch.stack(Post_mean), 0)
        # print(Post_mean2)

        self.attacker_optimizer.zero_grad()
        with torch.no_grad():
            i = 0
            for param in self.attacker_policy_network.parameters():
                tochange = param.data.flatten()
                aux = tochange.size()[0]

                prepare_grad = Post_mean2[i:aux + i, 0]

                i += aux
                tochange += self.config.alpha_attacker * prepare_grad

                param.data = torch.clone(tochange.reshape(param.data.shape))


        # Compute gradient and update models
        if attacker:

            # reset gradients
            #self.attacker_optimizer.zero_grad()

            # expected loss over the batch
            policy_loss_total = torch.stack(policy_loss).sum()
            policy_loss = policy_loss_total/num_batches


            # perform backprop
            #policy_loss.backward()

            # maybe clip gradient
            #if self.config.clip_gradient:
            #    torch.nn.utils.clip_grad_norm_(self.attacker_policy_network.parameters(), 1)

            # gradient descent step
            #self.attacker_optimizer.step()


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

        #print('my attacker state')
        #print(attacker_state)

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

        nu00 = 0.1 # 0.01


        # Training
        for iter in range(self.config.num_episodes):
            nu0 = nu00
            flag0 = True
            M = self.M
            sig2 = 100

            Ginv = None



            saved_grads_batch = []
            post_mean_batch = []

            # Batch
            for episode in range(self.config.batch_size):

                R = []
                prior_accum = 0

                for m in range(M):

                    episode_attacker_reward = 0
                    episode_defender_reward = 0
                    episode_step = 0
                    episode_attacker_loss = 0.0
                    episode_defender_loss = 0.0
                    saved_attacker_log_probs = []
                    saved_attacker_rewards = []
                    saved_defender_log_probs = []
                    saved_defender_rewards = []

                    saved_grads = []

                    prior_rewards = []

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
                            attacker_action, attacker_log_prob, _grads = self.get_action(attacker_state, attacker=True,
                                                                                 legal_actions=legal_actions,
                                                                                 non_legal_actions=illegal_actions)

                            if self.env.local_view_features():
                                attacker_action = PolicyGradientAgent.convert_local_attacker_action_to_global(attacker_action, attacker_obs)
                            saved_attacker_log_probs.append(attacker_log_prob)




                        if self.config.defender:
                            defender_action, defender_log_prob, _grads = self.get_action(defender_state, attacker=False)
                            saved_defender_log_probs.append(defender_log_prob)



                        action = (attacker_action, defender_action)

                        # Set prior here
                        if self.prior_state:
                            if self.prior_type == 1:
                                prior_reward = self.env.prior(action)

                            if self.prior_type == 2:
                                prior_reward = self.env.prior2(action)


                        prior_rewards.append(prior_reward)

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

                        saved_grads.append(_grads)

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

                    # Accumulate batch
                    saved_attacker_log_probs_batch.append(saved_attacker_log_probs)
                    saved_attacker_rewards_batch.append(saved_attacker_rewards)
                    saved_defender_log_probs_batch.append(saved_defender_log_probs)
                    saved_defender_rewards_batch.append(saved_defender_rewards)

                    R.append(episode_attacker_reward)
                    saved_grads_batch.append(saved_grads)

                    '''
                    print('')
                    print('R', R)
                    print('saved_attacker_rewards', saved_attacker_rewards)
                    '''
                    '''
                    print('prior rewards', prior_rewards)
                    print('sum prior rewards', sum(prior_rewards))
                    print('reward', saved_attacker_rewards)
                    print('sum reward', sum(saved_attacker_rewards))
                    print('')
                    '''

                    if m == 0:
                        uz1 = torch.stack(saved_grads_batch[0], dim=0).sum(dim=0).reshape(-1, 1)
                        R1 = sum(saved_attacker_rewards_batch[-1])
                        prior = sum(prior_rewards)/len(prior_rewards)
                        Y = Variable(torch.Tensor([R1-prior]), requires_grad=True).reshape(1,1)
                        Z = Variable(uz1, requires_grad=True)
                        if torch.cuda.is_available() and self.config.gpu:
                            device = torch.device("cuda:" + str(self.config.gpu_id))
                            Y = Y.to(device)

                    else:
                        R1 = sum(saved_attacker_rewards_batch[-1])
                        prior = sum(prior_rewards)/len(prior_rewards)
                        uzj = torch.stack(saved_grads_batch[-1], dim=0).sum(dim=0).reshape(-1, 1)
                        auxY = Variable(torch.Tensor([R1-prior]), requires_grad=True).reshape(1, 1)

                        if torch.cuda.is_available() and self.config.gpu:
                            device = torch.device("cuda:" + str(self.config.gpu_id))
                            auxY = auxY.to(device)

                        Y = torch.cat((Y, auxY), 1)
                        Z = torch.cat((Z, uzj), 1)




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

                    prior_accum += prior*sum(saved_attacker_log_probs)

                Post_mean = prior_accum/M + (Z @ Y.T)
                post_mean_batch.append(Post_mean)

            # Decay LR after every iteration
            lr_attacker = self.config.alpha_attacker
            if self.config.lr_exp_decay:
                self.attacker_lr_decay.step()
                lr_attacker = self.attacker_lr_decay.get_lr()[0]





            # Perform Batch Policy Gradient updates
            if self.config.attacker:
                #loss = self.training_step(saved_attacker_rewards_batch, saved_attacker_log_probs_batch, attacker=True)
                loss = self.training_step(Post_mean=post_mean_batch, R=saved_attacker_rewards_batch, attacker=True, lr=lr_attacker, Fisher_inv=Ginv, log_probs=saved_attacker_log_probs_batch)
                episode_attacker_loss += loss.item()





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

        # Video config
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
                    attacker_action, _, _ = self.get_action(attacker_state, attacker=True,
                                                         legal_actions=legal_actions, non_legal_actions=illegal_actions)
                    if self.env.local_view_features():
                        attacker_action = PolicyGradientAgent.convert_local_attacker_action_to_global(attacker_action, attacker_obs)
                if self.config.defender:
                    defender_action, _, _ = self.get_action(defender_state, attacker=False)
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





    def Kernel_Online_vs2(self, last_grad=None, grad_dict=None,
                          Ginv=None, R=None, yM=None, zM=None, C_inv = None,
                          Kprev_inv=None, Kprev = None, sig2 = None, A = None,
                          nu = None, episode = None, max_episode = None, one = None, iter=None, batch=None):

        #print('')
        #print('R inside')
        #print(R)
        #print('')

        auxY = Variable(torch.Tensor([sum(R)]), requires_grad=True).reshape(1, 1)
        k = torch.zeros(len(grad_dict), 1)
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            k = k.to(device)
            auxY = auxY.to(device)

        uzj = torch.stack(last_grad, dim=0).sum(dim=0).reshape(-1, 1)
        for t in range(len(grad_dict)):
            uzi = torch.stack(grad_dict[t], dim=0).sum(dim=0).reshape(-1, 1)
            #print('hihi')
            #print(torch.linalg.norm(uzi - uzj) / (100))
            k[t, 0] = 1/(1 + torch.linalg.norm(uzi - uzj) / (10))
            #k[t, 0] = uzi.T @ uzj
            #print(k[t, 0])


        a = Kprev_inv @ k
        #ktt = uzj.T @ uzj
        ktt = 1/(1 + torch.linalg.norm(uzj - uzj) / 100)

        delta = ktt - k.T @ a
        self.tensorboard_writer.add_scalar('delta/train/attacker', delta, self.config.batch_size*iter*max_episode + episode + batch)

        if (delta > nu) or (iter % 20 == 0):
            flag_dictionary = True
            #aux = torch.Tensor([1]).to(self.device)

            K = torch.hstack([torch.vstack([Kprev, k.T]), torch.vstack([k, ktt])])
            Kinv = (1 / delta) * torch.hstack(
                [torch.vstack([(delta * Kprev_inv + a @ a.T), -a.T]), torch.vstack([-a, one])])

            a[-1, 0] = 1
            a[0:-1, 0] = 0

            zero = torch.zeros_like(k)
            zero_fill = torch.zeros_like(A[:, 0]).reshape(-1, 1)

            si = sig2 * one + ktt - k.T @ A.T @ C_inv @ A @ k
            gi = C_inv @ A @ k

            C_inv = (1 / si) * torch.hstack(
                [torch.vstack([(si * C_inv + gi @ gi.T), (-gi).T]), torch.vstack([-gi, one])])
            A = torch.hstack([torch.vstack([A, zero.T]), torch.vstack([zero_fill, one])])

        else:
            K = Kprev
            Kinv = Kprev_inv

            si = sig2 * one + a.T @ K @ a - (K @ a).T @ A.T @ C_inv @ A @ (K @ a)
            gi = C_inv @ A @ K @ a

            C_inv = (1 / si) * torch.hstack(
                [torch.vstack([(si * C_inv + gi @ gi.T), (-gi).T]), torch.vstack([-gi, one])])
            A = torch.vstack([A, a.T])

            flag_dictionary = False

        yM = torch.cat((yM, auxY), 1)
        zM = torch.cat((zM, uzj), 1)

        # prepare some outputs
        if episode == max_episode:
            Post_mean = (zM @ C_inv @ yM.T)

        else:
            Post_mean = None

        return Post_mean, K, Kinv, flag_dictionary, yM, zM, A, C_inv



'''
###https://github.com/wookayin/tensorflow-plot/blob/master/examples/summary_heatmap.py
def heatmap_overlay(data, overlay_image=None, cmap='jet',
                    cbar=False, show_axis=False, alpha=0.5, **kwargs):
    fig, ax = tfplot.subplots(figsize=(5, 4) if cbar else (4, 4))
    fig.subplots_adjust(0, 0, 1, 1)  # use tight layout (no margins)
    ax.axis('off')

    if overlay_image is None: alpha = 1.0
    sns.heatmap(data, ax=ax, alpha=alpha, cmap=cmap, cbar=cbar, **kwargs)

    if overlay_image is not None:
        h, w = data.shape
        ax.imshow(overlay_image, extent=[0, h, 0, w])

    if show_axis:
        ax.axis('on')
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    return fig
'''





























