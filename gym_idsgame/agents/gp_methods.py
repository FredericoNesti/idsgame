"""This file is to support with general functions the implementation of Gaussian Processes for the model-free RL
methods from gym_idsgame  library """

# TODO: incorporate methods to be computed and remain inside GPU


import numpy as np
# import torch




def H(gamma, V):
    H = np.zeros((V.shape[0], V.shape[1]))
    for i in range(V.shape[0] - 1):
        H[i, i] = 1
        H[i, (i + 1)] = -gamma
    H[-1, -1] = 1
    return torch.Tensor(H)


def Rprev(H, V, Sigma):
    N = torch.Tensor(np.random.multivariate_normal(mean=np.zeros((Sigma.shape[0])), cov=Sigma, size=1)).reshape(-1, 1)
    Rprev = torch.Tensor(H @ V.to('cpu') + N)
    return Rprev


def Sigma(sigma, H):
    Sigma = np.power(sigma, 2) * H @ H.T
    return Sigma


def U(u):
    U = u @ u.T
    return U


def uz(log_prob):
    uz = log_prob.reshape(-1, 1)
    return uz


def kf(u1, u2, G):
    assert kf.size == 1
    kf = u1.T @ torch.inverse(G) @ u2
    return kf


def kernel(t, U, G):
    K = np.zeros((t, t))  # can also be t x t+1
    ## define the arbitrary state kernel here:
    kx = 1
    # kf = U @ torch.inverse(G) @ U.T
    kf = U @ torch.inverse(G) @ U.T
    # print(kf.shape)
    for i in range(t):
        for j in range(t):
            ## define here the fisher kernel for z,z'
            if i == j:
                k = kx + kf[i, j]
            else:
                k = kf[i, j]
            K[i, j] += k

    return K













def GPTD_step(value, log_probs, gamma, sigma):
    """
    This is a Gaussian Process step to be done recursively whenever we want to compute an integral with the help of Bayes Quadrature method.
    The correspondent integral might be related to loss function to be computed and backpropagated or a gradient distribution
    """
    t = len(value)

    u = BayesActorCriticAgent.uz(torch.stack(log_probs))

    U = BayesActorCriticAgent.U(u)

    Ghat = (1 / (t + 1)) * U @ U.T

    H = BayesActorCriticAgent.H(gamma, Ghat)
    Sigma = BayesActorCriticAgent.Sigma(sigma, H)
    Rprev = BayesActorCriticAgent.Rprev(H, Ghat, Sigma)
    # Rprev = Rprev.to(torch.device("cuda:" + str(self.config.gpu_id)))

    K = torch.Tensor(BayesActorCriticAgent.kernel(t, U, Ghat))
    # K = K.to(torch.device("cuda:" + str(self.config.gpu_id)))

    aux = torch.inverse(H @ K @ H.T + Sigma)
    a = H.T @ aux @ Rprev
    # C = H.T @ aux @ H

    Qall = K.T @ a

    return a, Qall











def GPTD_step(sample_size, policy, prev_dict_paths = {}):
    """
    This is a Gaussian Process step to be done recursively whenever we want to compute an integral with the help of Bayes Quadrature method.
    The correspondent integral might be related to loss function to be computed and backpropagated or a gradient distribution
    """

    dict_paths = {}
    if len(prev_dict_paths) == 0:

    Fisher_Info =

    for i in range(sample_size):




    t = len(value)

    u = BayesActorCriticAgent.uz(torch.stack(log_probs))

    U = BayesActorCriticAgent.U(u)

    Ghat = (1 / (t + 1)) * U @ U.T

    H = BayesActorCriticAgent.H(gamma, Ghat)
    Sigma = BayesActorCriticAgent.Sigma(sigma, H)
    Rprev = BayesActorCriticAgent.Rprev(H, Ghat, Sigma)
    # Rprev = Rprev.to(torch.device("cuda:" + str(self.config.gpu_id)))

    K = torch.Tensor(BayesActorCriticAgent.kernel(t, U, Ghat))
    # K = K.to(torch.device("cuda:" + str(self.config.gpu_id)))

    aux = torch.inverse(H @ K @ H.T + Sigma)
    a = H.T @ aux @ Rprev
    # C = H.T @ aux @ H

    Qall = K.T @ a

    return a, Qall
