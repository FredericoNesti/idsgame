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


#seed0 = 670979600
#torch.manual_seed(seed0)

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

        #print(action)

        log_prob = policy_dist.log_prob(action)

        #print(log_prob)

        #log_prob.backward(retain_graph=True)
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


    #####################################################################################




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

            returns = torch.Tensor([0]*len(R))

            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                returns = torch.Tensor(returns).to(device)
            R1 = 0
            for i,r in enumerate(R):
                r1 = sum(r)
                R1 = r1 + self.config.gamma * R1
                returns[-(i+1)] += R1


            if len(R) > 1:
                returns = ( returns - returns.mean() ) / ( returns.std() + self.machine_eps )


            #policy_loss0 = self.BPG_step(X, logP, returns, Ginv, model_type, Cinv)
            loss = - Post_mean * returns #- sum(log_probs[batch])/len(log_probs[batch])
            # posso implementar gradiente direto
            policy_loss.append(loss)


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


            # ALTERNATIVE COMPUTATION OF GRADIENTS
            self.attacker_optimizer.zero_grad()
            with torch.no_grad():
                i = 0
                for param in self.attacker_policy_network.parameters():
                    tochange = param.data.flatten()
                    aux = tochange.size()[0]

                    #print(aux)
                    #print(i)

                    prepare_grad = Post_mean[i:aux+i,0]

                    #print(prepare_grad)

                    i += aux
                    tochange += self.config.alpha_attacker*prepare_grad

                    param.data = torch.clone(tochange.reshape(param.data.shape))


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

        #nu00 = 0.00001
        nu00 = 0.001
        #nu_chunk = self.config.num_episodes / 100


        # Training
        #print('NUm epidsodes', self.config.num_episodes)
        for iter in range(self.config.num_episodes):
            #print('episode', iter)
            ### For Dictionary Update
            #nu0 = nu00 + 0.00000000000000001 * (iter / nu_chunk)
            nu0 = nu00
            flag0 = True
            M = 70
            sig2 = 100

            Ginv = None

            R = []
            # Batch
            #for episode in range(self.config.batch_size):

            saved_grads_batch = []

            for m in range(M):
                #dlogp = []
                #a = []
                #x = []

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

                    '''
                    print('')
                    print('CHECK OVER GRADS')
                    print(_grads)
                    '''
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


                #print(saved_attacker_rewards)
                #R.append(saved_attacker_rewards)
                #print(R)


                #saved_grads_batch.append(saved_grads)

                #Ginv = self.Var_params()


                #Fisher = self.Fisher_Info(saved_grads)
                #print('check Fisher', Fisher)

                #Ginv = torch.pinverse(Fisher)
                #print('check Ginv', Ginv)




                ### First Dictionary Update
                if flag0:
                    flag0 = False
                    # R
                    if self.config.attacker: R.append(episode_attacker_reward)
                    saved_grads_batch.append(saved_grads)

                    uz1 = torch.stack(saved_grads_batch[0], dim=0).sum(dim=0).reshape(-1, 1)

                    #Ginv = torch.inverse(uz1 @ uz1.T + 100*torch.eye(n=uz1.shape[0], m=uz1.shape[0], device=uz1.device)) #+ 100*torch.eye(n=uz1.shape[0], m=uz1.shape[0], device=uz1.device)

                    #print(Ginv.shape)
                    #print('Ginv check', torch.symeig(Ginv, eigenvectors=False))

                    #k11 = torch.pow(1 + uz1.T @ Ginv @ uz1, 2)
                    #k11 = torch.pow(1 + uz1.T @ uz1, 2)
                    #k11 = uz1.T @ uz1
                    k11 = 1/(1 + torch.linalg.norm(uz1-uz1)/100)

                    #k11 = uz1.T @ Ginv @ uz1
                    K = Variable(torch.Tensor([k11]), requires_grad=True).reshape(1, 1)
                    Kinv = Variable(torch.Tensor([1/k11]), requires_grad=True).reshape(1, 1)

                    #Y = Variable((R[-1] * uz1), requires_grad=True)
                    #Z = Variable((1 + uz1.T @ uz1), requires_grad=True)
                    Y = Variable(torch.Tensor([R[-1]]), requires_grad=True).reshape(1,1)
                    Z = Variable(uz1, requires_grad=True)

                    one = torch.Tensor([1]).reshape(1, 1)

                    if torch.cuda.is_available() and self.config.gpu:
                        device = torch.device("cuda:" + str(self.config.gpu_id))
                        Kinv = Kinv.to(device)
                        K = K.to(device)
                        one = one.to(device)
                        Y = Y.to(device)

                    Cinv = Variable(1 / (k11 + sig2 * one), requires_grad=True)
                    A = Variable(one, requires_grad=True)



                if m > 0:

                    Post_mean, K, Kinv, flag, Y, Z, A, Cinv = self.Kernel_Online_vs2(last_grad=saved_grads, grad_dict=saved_grads_batch,
                                                                          Ginv=Ginv, R=R, Kprev=K, Kprev_inv=Kinv, A = A, C_inv = Cinv, sig2=sig2,
                                                                          episode=m, max_episode=M - 1, nu=nu0, yM=Y, zM=Z, one=one, iter=iter)


                    #print(flag)
                    ### First Dictionary Update
                    if flag:
                        flag0 = False
                        if self.config.attacker: R.append(episode_attacker_reward)
                        saved_grads_batch.append(saved_grads)

                        #uzm = torch.stack(saved_grads_batch[-1], dim=0).sum(dim=0).reshape(-1, 1)
                        #m = len(saved_grads_batch)

                        #print('second flag', uzm.T @ Ginv @ uzm)

                        #Ginv = (1 / (1 - 1 / m)) * (Ginv - (1 / m) * (
                        #            (Ginv @ uzm @ (Ginv @ uzm).T) / (1 - (1 / m) + (1 / m) * uzm.T @ Ginv @ uzm)))






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

            #Ginv = self.Var_params()

            #K, Y, Z = self.Kernel_Y_Z(grad_dict=saved_grads_batch, Ginv=Ginv, R=saved_attacker_rewards_batch)

            #print(K)

            #Sigma = Variable(100 * torch.eye(n=K.shape[0], m=K.shape[1]), requires_grad=True)
            #if torch.cuda.is_available() and self.config.gpu:
            #    device = torch.device("cuda:" + str(self.config.gpu_id))
            #    Sigma = Sigma.to(device)

            #C = torch.inverse(K + Sigma)

            #Post_mean = Y @ C @ Z

            # Once done saving rewards send it to cuda
            #if torch.cuda.is_available() and self.config.gpu:
            #    device = torch.device("cuda:" + str(self.config.gpu_id))
            #    R = torch.Tensor(R).to(device)



            # Decay LR after every iteration
            lr_attacker = self.config.alpha_attacker
            if self.config.lr_exp_decay:
                self.attacker_lr_decay.step()
                lr_attacker = self.attacker_lr_decay.get_lr()[0]





            # Perform Batch Policy Gradient updates
            if self.config.attacker:
                #loss = self.training_step(saved_attacker_rewards_batch, saved_attacker_log_probs_batch, attacker=True)
                loss = self.training_step(Post_mean=Post_mean, R=saved_attacker_rewards_batch, attacker=True, lr=lr_attacker, Fisher_inv=Ginv, log_probs=saved_attacker_log_probs_batch)
                episode_attacker_loss += loss.item()

            if self.config.defender:
                #loss = self.training_step(saved_defender_rewards_batch, saved_defender_log_probs_batch, attacker=False)
                #loss = self.training_step(X, logP, R, Ginv, Cinv=Cinv, attacker=False)
                loss = self.training_step(Post_mean=Post_mean, R=saved_attacker_rewards_batch, attacker=True, lr=lr_attacker, Fisher_inv=Ginv)
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

            '''
            # Decay LR after every iteration
            lr_attacker = self.config.alpha_attacker
            if self.config.lr_exp_decay:
                self.attacker_lr_decay.step()
                lr_attacker = self.attacker_lr_decay.get_lr()[0]
            '''


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



    def Fisher_Info(self, grad_dict):
        d = 0
        flag = False
        #print(len(grad_dict))
        for x in grad_dict:
            d += len(x)
            #print(x)
            uz = x.reshape(-1, 1)
            if flag == False:
                Fisherinfo = uz @ uz.T
            if flag == True:
                Fisherinfo += uz @ uz.T
                flag = True
        Fisherinfo /= d
        #if torch.count_nonzero(torch.diag(Fisherinfo)) < torch.diag(Fisherinfo).size()[0]:
        #    Fisherinfo += torch.eye(n=Fisherinfo.shape[0], m=Fisherinfo.shape[1], device=Fisherinfo.device)
        return Fisherinfo

    '''
    def Fisher_Info_Hessian(self, grad_dict):
        d = 0
        flag = False
        #print(len(grad_dict))
        for x in grad_dict:
            d += len(x)
            #print(x)
            uz = x.reshape(-1, 1)
            if flag == False:
                Fisherinfo = uz @ uz.T
            if flag == True:
                Fisherinfo += uz @ uz.T
                flag = True
        Fisherinfo /= d
        #if torch.count_nonzero(torch.diag(Fisherinfo)) < torch.diag(Fisherinfo).size()[0]:
        #    Fisherinfo += torch.eye(n=Fisherinfo.shape[0], m=Fisherinfo.shape[1], device=Fisherinfo.device)
        return Fisherinfo
    '''



    def Var_params(self):
        model = self.attacker_policy_network
        W = torch.Tensor([])
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            W = W.to(device)

        for param in model.parameters():
            #print('')
            #print(param.data.shape)
            W = torch.hstack((W, param.data.flatten()))

        W = W.reshape(-1,1)
        Var = W @ W.T

        #I = 10*torch.eye(n=Var.shape[0], m=Var.shape[1])
        #if torch.cuda.is_available() and self.config.gpu:
        #    device = torch.device("cuda:" + str(self.config.gpu_id))
        #    I = I.to(device)

        #print('Var',Var)
        #print('I', I)

        #print('ok')

        #print(Var)

        return Var





    def Kernel_Online_vs2(self, last_grad=None, grad_dict=None,
                          Ginv=None, R=None, yM=None, zM=None, C_inv = None,
                          Kprev_inv=None, Kprev = None, sig2 = None, A = None,
                          nu = None, episode = None, max_episode = None, one = None, iter=None):

        auxY = Variable(torch.Tensor([sum(R)]), requires_grad=True).reshape(1, 1)
        k = torch.zeros(len(grad_dict), 1)
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            k = k.to(device)
            auxY = auxY.to(device)

        uzj = torch.stack(last_grad, dim=0).sum(dim=0).reshape(-1, 1)
        for t in range(len(grad_dict)):
            uzi = torch.stack(grad_dict[t], dim=0).sum(dim=0).reshape(-1, 1)
            #k[t, 0] = torch.pow(1 + uzi.T @ Ginv @ uzj, 2) #* (model_type == 1) #+ uzi.T @ Ginv @ uzj * (model_type == 2)
            #k[t, 0] = torch.pow(1 + uzi.T @ uzj, 2)
            #k[t, 0] = uzi.T @ uzj

            k[t, 0] = 1/(1 + torch.linalg.norm(uzi - uzj) / (100))

            #k[t, 0] = uzi.T @ Ginv @ uzj
        #uzj2 = torch.stack(grad_dict[-1], dim=0).sum(dim=0).reshape(-1, 1)

        #print(Kprev.shape)
        ##print(k.shape)

        a = Kprev_inv @ k
        #ktt = torch.pow(1 + uzj.T @ Ginv @ uzj, 2)
        #ktt = torch.pow(1 + uzj.T @ uzj, 2)

        # ktt = uzj.T @ uzj
        ktt = 1/(1 + torch.linalg.norm(uzj - uzj) / 100)

        #print('check nan')
        #print(uzt)
        #print(Ginv)



        #ktt = uzt.T @ Ginv @ uzt

        #delta = torch.abs(ktt - k.T @ a)

        '''
        print('')
        print(Kprev)
        print(a)
        print(k.T)
        print(ktt)
        print('')
        '''

        delta = ktt - k.T @ a

        #print('pre delta')
        #print(ktt)
        #print(k.T)
        #print('a', a)


        #print('')
        #print('delta', delta)


        #print('Ginv', torch.symeig(Ginv, eigenvectors=False))
        #print('K', torch.symeig(Kprev, eigenvectors=False))

        '''
        if self.config.attacker:
            for tag, value in self.attacker_policy_network.named_parameters():
                tag = tag.replace('.', '/')
                self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), iter)
                self.tensorboard_writer.add_histogram(tag + '_attacker/grad', value.grad.data.cpu().numpy(),
                                                      iter)
        '''
        self.tensorboard_writer.add_scalar('delta/train/attacker', delta, iter*max_episode + episode)

        if (delta > nu) or (iter % 50 == 0):
            flag_dictionary = True
            aux = torch.Tensor([1]).to(self.device)

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

        #yM = torch.cat((yM, (sum(R) * uzj)), 1)
        yM = torch.cat((yM, auxY), 1)

        #zM = torch.cat((zM, (1 + uzj.T @ Ginv @ uzj)), 1)
        #zM = torch.cat((zM, (1 + uzj.T @ uzj)), 1)
        zM = torch.cat((zM, uzj), 1)

        # prepare some outputs
        if episode == max_episode:
            #Post_mean = (yM @ C_inv @ zM.T)
            Post_mean = (zM @ C_inv @ yM.T)

        else:
            Post_mean = None

        return Post_mean, K, Kinv, flag_dictionary, yM, zM, A, C_inv








    def Kernel_Y_Z(self, grad_dict, Ginv, R):

        K = torch.zeros(size=(len(grad_dict), len(grad_dict)))
        Y = torch.zeros(size=(len(grad_dict[0][0]), len(grad_dict)))
        Z = torch.zeros(size=(len(grad_dict), 1))
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            K = K.to(device)
            Y = Y.to(device)
            Z = Z.to(device)

        for ti in range(len(grad_dict)):
            uzi = torch.stack(grad_dict[ti], dim=0).sum(dim=0).reshape(-1, 1)

            R1 = torch.Tensor(R[ti])
            if torch.cuda.is_available() and self.config.gpu:
                device = torch.device("cuda:" + str(self.config.gpu_id))
                R1 = R1.to(device)

            Y[:, ti] = R1.sum(dim=0).reshape(1,1) * uzi[:,0]
            Z[ti, :] = 1 + uzi.T @ Ginv @ uzi
            for tj in range(len(grad_dict)):
                uzj = torch.stack(grad_dict[tj], dim=0).sum(dim=0).reshape(-1, 1)
                K[ti, tj] = torch.pow(1 + uzi.T @ Ginv @ uzj, 2)



        return K, Y, Z




























