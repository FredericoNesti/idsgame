
#### LINEAR QUADRACTIC REGULATOR

import torch
from math import sqrt
#import argparse
import numpy as np
import random

seed0 = 670979600
torch.manual_seed(seed0)


num_episodes = 100

path_length = 20
estimate_comp_len = 40


beta01 = 0.3
beta02 = 0.6

sig2_r = 0.1

#####################################################################################

x0 = torch.normal(0.3, sqrt(0.001), size=(1,1))

def rewards(xt, at):
    nr = torch.normal(0., sqrt(sig2_r), size=(1,1))
    rt = xt**2 + 0.1*(at**2) + nr
    return rt

def transitions(xt, at):
    nx = torch.normal(0., sqrt(0.01), size=(1,1))
    xpos = xt + at + nx
    return xpos

def get_action(xt, theta_lambda, theta_sigma2):
    at = torch.normal(theta_lambda*xt, np.sqrt(theta_sigma2))
    return at

def sample_path(xt, theta_lambda, theta_sigma2):
    Rewards = []
    state_action_pair = []

    for i in range(path_length):
        at = get_action(xt, theta_lambda, theta_sigma2)
        #xt = transitions(xt, at)
        state_action_pair.append([xt[0,0], at[0,0]])
        rt = rewards(xt[0,0], at[0,0])
        Rewards.append(rt)
        xt = transitions(xt, at)
    return Rewards, state_action_pair



def train_LQR():

    k1 = -0.16
    k2 = 100

    # INIT
    theta_lambda = -1.999 + 1.998/(1 + np.exp(k1))
    theta_sigma2 = 0.001 + 1/(1 + np.exp(k2))

    for iter in range(num_episodes):

        betaj1 = beta01 * (20/(20 + iter))
        betaj2 = beta02 * (20 / (20 + iter))

        print('Iter % {}, Params {}'.format(iter/num_episodes, (theta_lambda, theta_sigma2)))

        flag0 = True
        flag = False
        Y = None
        Z = None

        path_dict_go = []
        reward_dict_go = []
        path_dict = []
        reward_dict = []

        d = 0
        Fisher = None

        for m1 in range(estimate_comp_len):
            R_m, sa_m = sample_path(xt=x0, theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            path_dict.append(sa_m)
            reward_dict.append(R_m)

            ue = score_func(sa_m, theta_lambda, theta_sigma2)
            if m1 == 0:
                Fisher = ue @ ue.T

            else:
                Fisher += ue @ ue.T

            d += len(sa_m)
        #print(d)
        #Fisher = Fisher/d


        for _ in range(500):
            R_m, sa_m = sample_path(xt=x0, theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            ue = score_func(sa_m, theta_lambda, theta_sigma2)
            Fisher += ue @ ue.T
            d += len(sa_m)

        Fisher = Fisher / (d*(estimate_comp_len+500))


        '''
        print('')
        print('PATH', path_dict)
        #print('REWARD', reward_dict)
        print('')
        print('FISHER', Fisher)
        print('')
        '''

        Ginv = torch.inverse(Fisher)

        #print('Ginv', Ginv)


        #Ginv[0,0] = 1
        Ginv[0,1] = 0
        Ginv[1,0] = 0
        #Ginv[1,1] = 2


        #print('Ginv', Ginv)
        #print('')

        #Ginv = torch.diag(torch.Tensor([theta_sigma2, 2*theta_sigma2**2]))

        for m0 in range(estimate_comp_len):

            R_m = reward_dict[m0]
            sa_m = path_dict[m0]

            ### First Dictionary Update
            if flag0:
                flag0 = False

                path_dict_go.append(sa_m)
                reward_dict_go.append(R_m)



                uz1 = score_func(path=path_dict[0], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
                #k11 = uz1.T @ Ginv @ uz1
                #k11 = uz1.T @ uz1
                k11 = torch.pow(1 + uz1.T @ uz1, 2)
                #fisher_ker_aux = torch.exp(-torch.linalg.norm(uz1 - uz1) / (2)).reshape(1,1)

                #fisher_ker_aux = -torch.log(1 + torch.linalg.norm(uz1 - uz1)).reshape(1, 1)

                #k11 = torch.pow(1 + fisher_ker_aux, 2)

                K = torch.Tensor([k11/k11]).reshape(1, 1)

                #Y = sum(R_m)/len(R_m)
                #Z = uz1

                Y = sum(R_m)*uz1
                Z = 1 + uz1.T @ uz1
                #Z = 1 + fisher_ker_aux


            if m0 > 0:
                Post_mean, K, flag, Y, Z = Kernel_Online(last_path=sa_m, path_dict=path_dict_go, Kprev=K, Ginv=Ginv,
                                                        R=R_m, yM=Y, zM=Z, theta_lambda=theta_lambda,
                                                        theta_sigma2=theta_sigma2)

            if flag:
                flag0 = False
                path_dict_go.append(sa_m)
                reward_dict_go.append(R_m)


        #K, Z = True_Kernel(path_dict=path_dict, Ginv=Ginv, theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)

        #for i in range(len(reward_dict)):
        #    if i == 0:
        #        Y = sum(reward_dict[i])
        #    else:
        #        Y = torch.cat((Y, sum(reward_dict[i])), 1)



        Sigma1 = 0.5 * torch.eye(n=K.shape[0], m=K.shape[1])
        Sigma2 = 0.5 * torch.eye(n=K.shape[0], m=K.shape[1])

        Sigma1 = sig2_r*estimate_comp_len*Sigma1
        Sigma2 = sig2_r * estimate_comp_len * Sigma2

        Sigma = Sigma1 + Sigma2
        Cinv = torch.inverse(K + Sigma)


        print('')
        print('Z', Z)
        print('Cinv', Cinv)
        print('Kernel', K)
        #print('Sigma', Sigma)
        print('Y', Y)
        print('')


        #Post_mean = Z @ Cinv @ Y.T
        Post_mean = Y @ Cinv @ Z.T

        print(Post_mean)

        theta_lambda -= betaj1 * Post_mean[0,:]
        theta_sigma2 -= betaj2 * Post_mean[1,:]

    return 0


def score_func_base(xi, ai, theta_lambda, theta_sigma2):
    return torch.Tensor([xi*(ai - theta_lambda * xi) / theta_sigma2,
                         (1/np.sqrt(theta_sigma2)**3)*((ai - theta_lambda*xi)**2) - (1/np.sqrt(theta_sigma2))]).reshape(-1, 1)



def score_func(path, theta_lambda, theta_sigma2):
    i = 0
    for step in path:
        if i == 0:
            uz = score_func_base(xi=step[0], ai=step[1], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            i = 1
        else:
            uz += score_func_base(xi=step[0], ai=step[1], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
    return uz


def score_func2(path, theta_lambda, theta_sigma2):

    path_torch = torch.Tensor(path)
    path2_torch = torch.pow(torch.Tensor(path), 2)

    x_mean = torch.mean(path_torch, dim=0)[0]
    a_mean = torch.mean(path_torch, dim=0)[1]

    x2_mean = torch.mean(path2_torch, dim=0)[0]
    a2_mean = torch.mean(path2_torch, dim=0)[1]

    aux1 = (1/theta_sigma2)*(len(path)*a_mean*x_mean - theta_lambda*len(path)*x2_mean)
    aux2 = -len(path)*(1/np.sqrt(theta_sigma2)) \
           + (1/np.sqrt(theta_sigma2)**3)*len(path)*(a2_mean - 2*theta_lambda*a_mean*x_mean + theta_lambda*x2_mean)

    #uz = torch.Tensor([aux1, aux2]).reshape(-1, 1)
    uz = torch.Tensor([a_mean, a2_mean -1]).reshape(-1, 1)
    return uz



def Kernel_Online(last_path, path_dict, Kprev, Ginv, R, theta_lambda, theta_sigma2, yM, zM):

    k = torch.zeros(len(path_dict), 1)

    uzj = score_func(path=last_path, theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
    for t in range(len(path_dict)):
        uzi = score_func(path=path_dict[t], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
        #k[t, 0] = uzi.T @ Ginv @ uzj

        #fisher_ker = torch.exp(-torch.linalg.norm(uzj-uzi)/(2*1000000))
        #fisher_ker = -torch.log(1 + torch.linalg.norm(uzj - uzi)/0.0000000000001)

        kii = torch.pow(1 + uzi.T @ uzi, 2)
        kjj = torch.pow(1 + uzj.T @ uzj, 2)
        k[t, 0] = torch.pow(1 + uzi.T @ uzj, 2) / torch.sqrt(kii*kjj)
        #k[t, 0] = torch.pow(1 + fisher_ker, 2)


    #ktt = uzj.T @ Ginv @ uzj

    ktt = torch.pow(1 + uzj.T @ uzj, 2)
    #fisher_ker_aux = torch.exp(-torch.linalg.norm(uzj - uzj) / 2).reshape(1,1)
    #fisher_ker_aux = -torch.log(1+torch.linalg.norm(uzj - uzj)).reshape(1, 1)

    #ktt = torch.pow(1 + fisher_ker_aux, 2)

    flag_dictionary = True

    K = torch.hstack([torch.vstack([Kprev, k.T]), torch.vstack([k, ktt/ktt])])

    #yM = torch.cat((yM, sum(R)/len(R)), 1)
    #zM = torch.cat((zM, uzj), 1)

    yM = torch.cat((yM, sum(R)*uzj), 1)
    zM = torch.cat((zM, 1 + (uzj.T @ uzj)/torch.linalg.norm(uzj)), 1)
    #zM = torch.cat((zM, 1 + fisher_ker_aux), 1)


    Post_mean = None
    return Post_mean, K, flag_dictionary, yM, zM


def True_Kernel(path_dict, Ginv, theta_lambda, theta_sigma2):

    K = torch.zeros(len(path_dict), len(path_dict))

    for i in range(len(path_dict)):
        uzi = score_func(path=path_dict[i], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
        for j in range(len(path_dict)):
            uzj = score_func(path=path_dict[j], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            K[i, j] = uzi.T @ Ginv @ uzj
        if i == 0:
            Z = uzi
        else:
            Z = torch.cat((Z, uzi),1)


    return K, Z


def Fisher_Info(xi, ai, theta_lambda, theta_sigma2):
    I = torch.zeros((2,2))

    a = (xi**2)/theta_sigma2
    b = (1/torch.sqrt(theta_sigma2)**5)*xi*(ai - theta_lambda*xi)**3
    c = b
    d = (1/torch.sqrt(theta_sigma2)**6)*(ai - theta_lambda*xi)**4 + 1

    I[0,0] = a
    I[1, 0] = b
    I[0, 1] = c
    I[1, 1] = d

    return I


def Fisher_Info_Inv(xi, ai, theta_lambda, theta_sigma2):
    I = torch.zeros((2, 2))

    print(xi)
    print(ai)

    a = (xi ** 2) / theta_sigma2
    b = 0
    c = 0
    d = 2 / theta_sigma2

    det = a*d - b*c

    I[0, 0] = d
    I[1, 0] = -b
    I[0, 1] = -c
    I[1, 1] = a

    return I/det


#######################################################################################

if __name__ == '__main__':
    #seed0 = 1000
    #torch.manual_seed(seed0)
    #np.random.seed(seed0)
    #random.seed(seed0)

    train_LQR()
