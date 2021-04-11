
#### LINEAR QUADRACTIC REGULATOR

import torch
from math import sqrt
#import argparse
import numpy as np

### DEFINE PARAMETERS HERE

seed1 = 347933
seed2 = 124650
seed3 = 102824

num_episodes = 100

path_length = 20
estimate_comp_len = 40 #10000

sig2 = 100

### system parameters

nu0 = 0.1

beta01 = 0.
beta02 = 0.

sig2_r = 0.1

###
trained_lambda = None
trained_sigma2 = None

###print

x0 = torch.normal(0.3, sqrt(0.001), size=(1,1))

#####################################################################################

def rewards(xt, at):
    nr = torch.normal(0., sqrt(sig2_r), size=(1,1))
    rt = xt**2 + 0.1*at**2 + nr
    return rt

def transitions(xt, at):
    nx = torch.normal(0., sqrt(0.01), size=(1,1))
    xpos = xt + at + nx
    return xpos

def get_action(xt, theta_lambda, theta_sigma2):
    at = torch.normal(theta_lambda*xt, sqrt(theta_sigma2))
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
        Kinv = None
        Y = None
        Z = None

        path_dict_go = []
        reward_dict_go = []


        super_flag = True


        #for d in range(200):
        path_dict = []
        reward_dict = []
        diag_Sigma1 = []
        diag_Sigma2 = []

        for m0 in range(estimate_comp_len):
            R_m, sa_m = sample_path(xt=x0, theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            path_dict.append(sa_m)
            reward_dict.append(R_m)

            '''
            if m0 == 0:
                uz1 = score_func(path=path_dict[0], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
                #Fisher = uz1 @ uz1.T

                #diag_Sigma1.append(uz1[0,:]**2)
                #diag_Sigma2.append(uz1[1, :]**2)

            else:
                uz1 = score_func(path=path_dict[m0], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
                #Fisher += uz1 @ uz1.T

                #diag_Sigma1.append(uz1[0,:]**2)
                #diag_Sigma2.append(uz1[1, :]**2)
            '''

        #Fisher /= estimate_comp_len

        #Sigma1 = torch.diag(torch.Tensor(diag_Sigma1))
        #Sigma2 = torch.diag(torch.Tensor(diag_Sigma2))

        #Ginv = torch.inverse(Fisher)

        Ginv = torch.diag(torch.Tensor([theta_sigma2, 2*theta_sigma2**2]))


        for m0 in range(estimate_comp_len):

            R_m = reward_dict[m0]
            sa_m = path_dict[m0]
            #R_m, sa_m = sample_path(xt=x0, theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            #s_m, a_m = sa_m

            ### First Dictionary Update
            if flag0:
                flag0 = False

                path_dict_go.append(sa_m)
                reward_dict_go.append(R_m)

                #Fisher = Fisher_Info(path_dict, theta_lambda, theta_sigma2)

                uz1 = score_func(path=path_dict[0], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
                #Fisher = uz1 @ uz1.T
                #Ginv = torch.inverse(Fisher) #+ torch.eye(n=Fisher.shape[0], m=Fisher.shape[1])

                #k11 = torch.pow(1 + uz1.T @ Ginv @ uz1, 2)
                k11 = uz1.T @ Ginv @ uz1

                K = torch.Tensor([k11]).reshape(1, 1)
                Kinv = torch.Tensor([1/k11]).reshape(1, 1)

                #Y = sum(R_m) * uz1
                #Z = 1 + uz1.T @ Ginv @ uz1

                Y = sum(R_m)/len(R_m)
                Z = uz1

                one = torch.Tensor([1]).reshape(1, 1)

                Cinv = 1/(k11 + sig2*one)
                A = one


            #print('')
            #print('')
            #print('m', m)

            #print('Cinv', Cinv)

            #print('A', A)

            #print('')
            #print('Kernel', K)

            #print('rewrad dict', reward_dict_go)

            if m0 > 0:
                Post_mean, K, Kinv, flag, Y, Z, A, Cinv = Kernel_Online(last_path=sa_m, path_dict=path_dict_go,
                                                                 Kprev=K, Ginv=Ginv, A = A, C_inv=Cinv,
                                                                 R=R_m, yM=Y, zM=Z, Kprev_inv=Kinv,
                                                                 sigma2=sig2, episode=m0, max_episode=estimate_comp_len-1,
                                                                 theta_lambda=theta_lambda, theta_sigma2=theta_sigma2, iter = iter)

            #print('eig Ginv', torch.symeig(Ginv, eigenvectors=False))
            #print('eig Kinv', torch.symeig(Kinv, eigenvectors=False))


            if flag:
                flag0 = False
                path_dict_go.append(sa_m)
                reward_dict_go.append(R_m)

                #uzm = score_func(path=path_dict[m0], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)

                #m = len(path_dict) #m0 + 1

                #Fisher = Fisher_Info(path_dict)
                #Ginv = torch.pinverse(Fisher)

                #Ginv = (1/(1 - 1/m)) * (Ginv - (1/m)*((Ginv @ uzm @ (Ginv @ uzm).T)/(1 - (1/m) + (1/m)*uzm.T @ Ginv @ uzm)))


        del Cinv, Post_mean#, Sigma1, Sigma2

        Sigma1 = 0.5 * torch.eye(n=K.shape[0], m=K.shape[1])
        Sigma2 = 0.5 * torch.eye(n=K.shape[0], m=K.shape[1])

        Sigma1 = sig2_r*estimate_comp_len*Sigma1
        Sigma2 = sig2_r * estimate_comp_len * Sigma2



        #Cinv1 = torch.inverse(K + Sigma1) # fix sigma
        #Cinv2 = torch.inverse(K + Sigma2) # fix sigma

        #print(Y[1, :])
        #print(Cinv2)
        #print(Z.T)

        #Post_mean1 = Y[0,:] @ Cinv1 @ Z.T
        #Post_mean2 = Y[1, :] @ Cinv2 @ Z.T

        '''
        if Post_mean1 <= -2.:
            Post_mean1 = -2.
        if Post_mean1 >= 2.:
            Post_mean1 = 2.

        if Post_mean2 <= -2.:
            Post_mean2 = -2.
        if Post_mean2 >= 2.:
            Post_mean2 = 2.
        '''

        #print(Post_mean1)
        #print(Post_mean2)


        Sigma = Sigma1 + Sigma2
        Cinv = torch.inverse(K + Sigma)
        #Post_mean = Y @ Cinv @ Z.T
        Post_mean = Z @ Cinv @ Y.T

        print(Post_mean)

        theta_lambda += betaj1 * Post_mean[0,:]
        theta_sigma2 += betaj2 * Post_mean[1,:]

        if theta_sigma2 <= 0.:
            print('yes')
            theta_sigma2 = 0.001

        #print('Kernel', K)


    return 0


def score_func_base(xi, ai, theta_lambda, theta_sigma2):
    #return torch.Tensor([ai - theta_lambda*xi, ai**2 - (theta_lambda*xi)**2 - theta_sigma2]).reshape(-1,1)
    return torch.Tensor([(ai - theta_lambda * xi)/theta_sigma2,
                         (1/(2*theta_sigma2))*((ai - theta_lambda * xi)**2/theta_sigma2 - 3)]).reshape(-1, 1)

def score_func(path, theta_lambda, theta_sigma2):
    i = 0
    for step in path:
        if i == 0:
            uz = score_func_base(xi=step[0], ai=step[1], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            i = 1
        else:
            uz += score_func_base(xi=step[0], ai=step[1], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
    return uz

def Fisher_Info(path_dict, theta_lambda, theta_sigma2):
    d = 0
    flag = False
    for path in path_dict:
        d += 1
        for step in path:
            uz = score_func_base(xi=step[0], ai=step[1], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
            if flag == False:
                Fisherinfo = uz @ uz.T
            if flag == True:
                Fisherinfo += uz @ uz.T
                flag = True
    Fisherinfo /= d
    return Fisherinfo


def Kernel_Online(last_path, path_dict, Kprev, Kprev_inv, Ginv, R, theta_lambda, theta_sigma2, A, C_inv, iter,
                  yM, zM, sigma2=100, episode=None, max_episode=None):

    k = torch.zeros(len(path_dict), 1)

    uzj = score_func(path=last_path, theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
    for t in range(len(path_dict)):
        uzi = score_func(path=path_dict[t], theta_lambda=theta_lambda, theta_sigma2=theta_sigma2)
        #k[t, 0] = torch.pow(1 + uzi.T @ Ginv @ uzj, 2)
        k[t, 0] = uzi.T @ Ginv @ uzj

    #a = Kprev_inv @ k
    a = None

    #print('a', a)

    #aux = 1 + uzj.T @ Ginv @ uzj
    #ktt = torch.pow(aux, 2)

    ktt = uzj.T @ Ginv @ uzj

    #delta = ktt - k.T @ a
    delta = -100000

    #print('delta', delta)

    #one = torch.Tensor([1]).reshape(1,1)

    #if delta > nu0 or iter == 0 or super_flag:
    flag_dictionary = True

    K = torch.hstack([torch.vstack([Kprev, k.T]), torch.vstack([k, ktt])])
    #Kinv = (1/delta) * torch.hstack([torch.vstack([(delta * Kprev_inv + a @ a.T), -a.T]), torch.vstack([-a, one])])

    #a[-1,0] = 1
    #a[0:-1,0] = 0

    #zero = torch.zeros_like(k)
    #zero_fill = torch.zeros_like(A[:,0]).reshape(-1,1)

    #si = sig2*one + ktt - k.T @ A.T @ C_inv @ A @ k
    #gi = C_inv @ A @ k
    #print('si', si)

    #C_inv = (1/si)*torch.hstack([torch.vstack([(si*C_inv + gi @ gi.T), (-gi).T]), torch.vstack([-gi, one])])
    #A = torch.hstack([torch.vstack([A, zero.T]), torch.vstack([zero_fill, one])])
    A = None
    Kinv = None
    '''
    else:

        K = Kprev
        Kinv = Kprev_inv

        si = sig2*one + a.T @ K @ a - (K @ a).T @ A.T @ C_inv @ A @ (K @ a)
        gi = C_inv @ A @ K @ a

        #print('si', si)

        C_inv = (1/si)*torch.hstack([torch.vstack([(si*C_inv + gi @ gi.T), (-gi).T]), torch.vstack([-gi, one])])
        A = torch.vstack([A, a.T])

        flag_dictionary = False
    '''

    #yM = torch.cat((yM, (sum(R) * uzj)), 1)
    #zM = torch.cat((zM, aux), 1)

    yM = torch.cat((yM, sum(R)/len(R)), 1)
    zM = torch.cat((zM, uzj), 1)


    # prepare some outputs
    '''
    if episode == max_episode:
        Post_mean = (yM @ C_inv @ zM.T)

    else:
        Post_mean = None
    '''
    Post_mean = None
    return Post_mean, K, Kinv, flag_dictionary, yM, zM, A, C_inv








#######################################################################################

if __name__ == '__main__':
    #R, state_action_pair = sample_path()
    train_LQR()

    '''
    print('')
    print('Rewards')
    print(R)
    print('')
    print('')
    print('sampled path')
    print(state_action_pair)
    print('')
    '''
