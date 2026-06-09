'''
@Project ：DQN_IRL
@File    ：env.py
@Author  ：TXC
@Date    ：2021/7/29 9:26
'''

import os
import random
import torch
import numpy as np
import numpy as np
import math
import torch
# import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, toeplitz
from scipy import integrate
import scipy
from scipy.linalg import sqrtm
from scipy.stats import norm
from src.config import *
# from tt import *
import os
import scipy.io as scio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class CF_RS_HI:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

        # random.seed(random_seed)
        # np.random.seed(random_seed)
        # torch.manual_seed(random_seed)
        self._max_episode_steps = max_episode_steps
        self.K = K  # UE number
        self.L = L  # AP number
        self.J = J  # sub_message number
        self.N = N  # antenna number
        self.Pmax = Pmax
        self.nbrOfRealizations = nbrOfRealizations
        self.tau_p = tau_p
        self.kappa = kappa

        self.observation_shape = state_num
        # self.action_space = power_set
        self.action_shape = action_num

    def pathloss_threeslope(self, dvec):
        d0 = 10  # meter
        d1 = 50  # meter
        L = 140.7151

        PL = np.zeros(len(dvec))

        for ind, d in enumerate(dvec):
            if d <= d0:
                PL[ind] = -L - 15 * np.log10(d1 / 1000) - 20 * np.log10(d0 / 1000)
            elif d <= d1:
                PL[ind] = -L - 15 * np.log10(d1 / 1000) - 20 * np.log10(d / 1000)
            else:
                PL[ind] = -L - 35 * np.log10(d / 1000)

        return PL

    def functionRlocalscattering(self, M, theta, ASDdeg, antennaSpacing=1 / 2, distribution='Gaussian'):
        ASD = ASDdeg * np.pi / 180
        firstRow = np.zeros((M, 1), dtype=np.complex128)

        for column in range(M):
            distance = antennaSpacing * (column)

            if distribution == 'Gaussian':
                def F(Delta):
                    phase = np.exp(1j * 2 * np.pi * distance * np.sin(theta + Delta))
                    gaussian = np.exp(-Delta ** 2 / (2 * ASD ** 2)) / (np.sqrt(2 * np.pi) * ASD)
                    return phase * gaussian

                a = -20 * ASD
                b = 20 * ASD

                real_part, _ = integrate.quad(lambda Delta: np.real(F(Delta)), a, b)

                imag_part, _ = integrate.quad(lambda Delta: np.imag(F(Delta)), a, b)

                firstRow[column] = real_part + 1j * imag_part
            elif distribution == 'Uniform':
                limits = np.sqrt(3) * ASD
                F = lambda Delta: np.exp(1j * 2 * np.pi * distance * np.sin(theta + Delta)) / (2 * limits)
                firstRow[column], _ = integrate.quad(F, -limits, limits)
            elif distribution == 'Laplace':
                LaplaceScale = ASD / np.sqrt(2)
                F = lambda Delta: np.exp(1j * 2 * np.pi * distance * np.sin(theta + Delta)) * np.exp(
                    -np.abs(Delta) / LaplaceScale) / (2 * LaplaceScale)
                firstRow[column], _ = integrate.quad(F, -20 * ASD, 20 * ASD)

        R = toeplitz(firstRow).T

        return R

    def functionChannelEstimates(self):

        p = self.Pmax

        # path = r'E:\HI_DRL2\src'

        H_Rayleigh_data = scio.loadmat(os.path.join(path, 'data/H_Rayleigh' + '_L_' + str(self.L) + '_setup_' + '.mat'))

        H_data = scio.loadmat(os.path.join(path, 'data/H' + '_L_' + str(self.L) + '_setup_' + '.mat'))

        H_Mean_data = scio.loadmat(os.path.join(path, 'data/HMean' + '_L_' + str(self.L) + '_setup_'+ '.mat'))

        HMeanx_data = scio.loadmat(os.path.join(path, 'data/HMeanx' + '_L_' + str(self.L) + '_setup_' + '.mat'))

        R_data = scio.loadmat(os.path.join(path, 'data/R' + '_L_' + str(self.L) + '_setup_'  + '.mat'))

        pilotIndex_data = scio.loadmat(os.path.join(path, 'data/pilotIndex' + '_L_' + str(self.L) + '_setup_'  + '.mat'))

        H_Rayleigh = H_Rayleigh_data['H_Rayleigh']
        H = H_data['H']
        HMean = H_Mean_data['HMean']
        HMeanx = HMeanx_data['HMeanx']
        R = R_data['R']
        pilotIndex = pilotIndex_data['pilotIndex'].reshape(-1)

        sigma2 = 1
        eyeN = np.eye(self.N)


        # Generate realizations of normalized noise
        Np = np.sqrt(0.5 * sigma2) * (
                np.random.randn(self.N, self.nbrOfRealizations, self.K, self.L) + 1j * np.random.randn(self.N, self.nbrOfRealizations, self.K, self.L))
        Eta = np.sqrt(0.5 * (1 - self.kappa) * p) * (
                np.random.randn(1, self.nbrOfRealizations, self.K) + 1j * np.random.randn(1, self.nbrOfRealizations, self.K))
        Etax = np.reshape(np.tile(Eta, (self.N, 1, 1)), (self.N, self.nbrOfRealizations, self.K))

        # Prepare to save
        Hhat = np.zeros((self.N, self.nbrOfRealizations, self.K, self.L), dtype=np.complex128)
        z = np.zeros((self.N, self.nbrOfRealizations, self.K, self.L), dtype=np.complex128)
        zbar = np.zeros((self.N, self.nbrOfRealizations, self.K, self.L), dtype=np.complex128)
        Qkl = np.zeros((self.N, self.N, self.K, self.L), dtype=np.complex128)
        Qkil = np.zeros((self.N, self.N, self.K, self.K, self.L), dtype=np.complex128)
        Hsum = np.zeros((self.N, self.N, self.K, self.L), dtype=np.complex128)
        Psi = np.zeros((self.N, self.N, self.K, self.L), dtype=np.complex128)

        for k in range(self.K):
            for l in range(self.L):
                z[:, :, k, l] = np.sqrt(self.kappa * p * self.tau_p) * np.sum(
                    H[:, :, np.where(pilotIndex[k] == pilotIndex)[0], l],
                    axis=2) + \
                                np.sum(H_Rayleigh[:, :, :, l] * Etax[:, :, :], axis=2) + \
                                np.sum(HMeanx[:, :, :, l] * Etax[:, :, :], axis=2) + Np[:, :, k, l]

                zbar[:, :, k, l] = np.sqrt(self.kappa * p * self.tau_p) * np.sum(
                    HMeanx[:, :, np.where(pilotIndex[k] == pilotIndex)[0], l], axis=2)

                for i in range(self.K):
                    Hsum[:, :, i, l] = np.outer(HMean[:, i, l], np.conj(HMean[:, i, l]))

                Psi[:, :, k, l] = self.kappa * p * self.tau_p * np.sum(R[:, :, np.where(pilotIndex[k] == pilotIndex)[0], l],
                                                             axis=2) + \
                                  (1 - self.kappa) * p * np.sum(R[:, :, :, l], axis=2) + \
                                  (1 - self.kappa) * p * np.sum(Hsum[:, :, :, l], axis=2) + sigma2 * eyeN

                Hhat[:, :, k, l] = HMeanx[:, :, k, l] + \
                                   np.sqrt(self.kappa * p * self.tau_p) * np.dot(R[:, :, k, l] / Psi[:, :, k, l],
                                                                       (z[:, :, k, l] - zbar[:, :, k, l]))

                Qkl[:, :, k, l] = self.kappa * self.tau_p * p * np.dot(np.linalg.solve(Psi[:, :, k, l].T, R[:, :, k, l].T).T,
                                                             R[:, :, k, l])

                for i in range(self.K):
                    Qkil[:, :, k, i, l] = self.kappa * p * self.tau_p * np.dot(
                        np.linalg.solve(Psi[:, :, k, l].T, R[:, :, i, l].T).T, R[:, :, k, l])

        b_k = np.zeros((self.K, self.L, self.J), dtype=np.complex128)
        A = np.zeros((self.L, self.L, self.K, self.J), dtype=np.complex128)
        Y = np.zeros((self.L, self.L, self.K, self.K, self.J), dtype=np.complex128)
        c = np.zeros((self.L, self.K, self.K, self.J), dtype=np.complex128)

        for l in range(self.L):
            for k in range(self.K):
                for j in range(self.J):
                    b_k[k, l, j] = np.dot(HMean[:, k, l].conj().T, HMean[:, k, l]) + np.trace(Qkl[:, :, k, l])
                    A[l, l, k, j] = np.dot(HMean[:, k, l].conj().T, HMean[:, k, l]) + np.trace(Qkl[:, :, k, l])
                    for i in range(self.K):
                        Y[l, l, k, i, j] = HMean[:, k, l].conj().T @ R[:, :, i, l] @ HMean[:, k, l] \
                                        + HMean[:, i, l].conj().T @ Qkl[:, :, k, l] @ HMean[:, i, l] \
                                        + np.trace(Qkl[:, :, k, l] @ R[:, :, i, l])
                        if i == k:
                            c[l, k, i, j] = 0
                        else:
                            if pilotIndex[i] == pilotIndex[k]:
                                c[l, k, i, j] = HMean[:, k, l].conj().T @ HMean[:, i, l] + np.trace(Qkil[:, :, k, i, l])
                            else:
                                c[l, k, i, j] = HMean[:, k, l].conj().T @ HMean[:, i, l]

        # path = r'E:\HI_DRL2\src'

        scio.savemat(os.path.join(path, "data\\b_k" + '_L_' + str(self.L) + '_setup_'  + '.mat'), {'b_k': b_k})
        scio.savemat(os.path.join(path, "data\\A" + '_L_' + str(self.L) + '_setup_'  + '.mat'), {'A': A})
        scio.savemat(os.path.join(path, "data\\Y" + '_L_' + str(self.L) + '_setup_'  + '.mat'), {'Y': Y})
        scio.savemat(os.path.join(path, "data\\c" + '_L_' + str(self.L) + '_setup_'  + '.mat'), {'c': c})
        scio.savemat(os.path.join(path, "data\\R" + '_L_' + str(self.L) + '_setup_' + '.mat'), {'R': R})

        return Hhat, Qkl, Qkil

    # return: 信道，方差，导频分配
    def Generate_H_set(self):
        # Size of the coverage area (as a square with wrap-around)
        squareLength = 500  # meter

        # Communication bandwidth
        B = 30e6

        # Noise figure (in dB)
        noiseFigure = 9

        # Compute noise power
        noiseVariancedBm = -174 + 10 * np.log10(B) + noiseFigure

        # Parameters for the shadow fading
        sigma_sf = 8
        delta = 0.5
        decorr = 100

        # Define the antenna spacing (in number of wavelengths)
        antennaSpacing = 1 / 2  # Half wavelength distance

        # Angular standard deviation around the nominal angle (measured in degrees)
        ASDdeg = 5

        # Prepare to save results
        R = np.zeros((self.N, self.N, self.L, self.K), dtype=complex)
        R_new = np.zeros((self.N, self.N, self.K, self.L), dtype=complex)
        distancesUE = np.zeros((self.L, self.K))
        gainOverNoisedB = np.zeros((self.L, self.K))
        ricianFactor = np.zeros((self.L, self.K))
        channelGain_LoS = np.zeros((self.L, self.K))
        channelGain_NLoS = np.zeros((self.L, self.K))
        HMean = np.zeros((self.N, self.L, self.K), dtype=complex)
        HMean_new = np.zeros((self.N, self.K, self.L), dtype=complex)

        # Generate random AP locations with uniform distribution
        APpositions = (np.random.rand(self.L) + 1j * np.random.rand(self.L)) * squareLength
        # Generate random UE locations with uniform distribution
        UEpositions = (np.random.rand(self.K) + 1j * np.random.rand(self.K)) * squareLength

        # Compute alternative AP locations by using wrap around
        wrapHorizontal = np.tile([-squareLength, 0, squareLength], (3, 1))
        wrapVertical = np.tile([-squareLength, 0, squareLength], (3, 1)).T
        wrapLocations = (wrapHorizontal + 1j * wrapVertical).flatten()
        APpositionsWrapped = np.tile(APpositions, (len(wrapLocations), 1)).T + np.tile(wrapLocations, (self.L, 1))
        UEpositionsWrapped = np.tile(UEpositions, (len(wrapLocations), 1)).T + np.tile(wrapLocations, (self.K, 1))

        # Compute the correlation matrices for the shadow fading
        shadowCorrMatrix_APs = np.zeros((self.L, self.L))
        shadowCorrMatrix_UEs = np.zeros((self.K, self.K))

        for l in range(self.L):
            distancetoAP = np.min(np.abs(APpositionsWrapped - np.tile(APpositions[l], APpositionsWrapped.shape)),
                                  axis=1)
            shadowCorrMatrix_APs[:, l] = 2 ** (-distancetoAP / decorr)

        for k in range(self.K):
            distancetoUE = np.min(np.abs(UEpositionsWrapped - np.tile(UEpositions[k], UEpositionsWrapped.shape)),
                                  axis=1)
            shadowCorrMatrix_UEs[:, k] = 2 ** (-distancetoUE / decorr)

        # Generate shadow fading realizations
        a = sigma_sf * np.dot(np.linalg.cholesky(shadowCorrMatrix_APs), np.random.randn(self.L))
        b = sigma_sf * np.dot(np.linalg.cholesky(shadowCorrMatrix_UEs), np.random.randn(self.K))
        q = scipy.linalg.sqrtm(shadowCorrMatrix_APs)

        for k in range(self.K):
            distancetoUE = np.min(np.abs(APpositionsWrapped - np.tile(UEpositions[k], APpositionsWrapped.shape)),
                                  axis=1)
            whichpos = np.argmin(np.abs(APpositionsWrapped - np.tile(UEpositions[k], APpositionsWrapped.shape)), axis=1)
            distancesUE[:, k] = distancetoUE

            gainOverNoisedB[:, k] = self.pathloss_threeslope(distancesUE[:, k]) - noiseVariancedBm

            gainOverNoisedB[distancetoUE > 50, k] += np.sqrt(delta) * a[distancetoUE > 50] + np.sqrt(1 - delta) * b[k]
            channelGain = 10 ** (gainOverNoisedB / 10)

            ricianFactor[:, k] = math.sqrt(10)

            channelGain_LoS[:, k] = ricianFactor[:, k] / (ricianFactor[:, k] + 1) * channelGain[:, k]
            channelGain_NLoS[:, k] = 1 / (ricianFactor[:, k] + 1) * channelGain[:, k]

            for l in range(self.L):
                angletoUE = np.angle(UEpositions[k] - APpositionsWrapped[l, whichpos[l]])
                correlationMatrix = self.functionRlocalscattering(self.N, angletoUE, ASDdeg, antennaSpacing)
                HMean_norm = np.exp(1j * 2 * np.pi * np.arange(self.N) * np.sin(angletoUE) * antennaSpacing)

                R[:, :, l, k] = channelGain_NLoS[l, k] * correlationMatrix
                HMean[:, l, k] = np.sqrt(channelGain_LoS[l, k]) * HMean_norm

        for k in range(self.K):
            for l in range(self.L):
                R_new[:, :, k, l] = R[:, :, l, k]
                HMean_new[:, k, l] = HMean[:, l, k]

        # Assign random pilots
        pilotIndex = np.mod(np.random.permutation(self.K), self.tau_p) + 1

        # Generate the channel realizations
        W = np.sqrt(0.5) * (
                    np.random.randn(self.N, self.nbrOfRealizations, self.K, self.L) + 1j * np.random.randn(self.N, self.nbrOfRealizations, self.K, self.L))

        H = np.zeros((self.N, self.nbrOfRealizations, self.K, self.L), dtype=complex)
        H_Rayleigh = np.zeros((self.N, self.nbrOfRealizations, self.K, self.L), dtype=complex)
        HMeanx = np.reshape(np.tile(HMean_new[:, :, :, np.newaxis], (1, 1, 1, self.nbrOfRealizations)),
                            (self.N, self.nbrOfRealizations, self.K, self.L))

        for l in range(self.L):
            for k in range(self.K):
                Rsqrt = scipy.linalg.sqrtm(R_new[:, :, k, l])
                H_Rayleigh[:, :, k, l] = Rsqrt @ W[:, :, k, l]
                H[:, :, k, l] = Rsqrt @ W[:, :, k, l] + HMeanx[:, :, k, l]

        # path = r'E:\HI_DRL2\src'

        scio.savemat(os.path.join(path, "data\\H_Rayleigh" + '_L_' + str(self.L) + '_setup_' + '.mat'), {'H_Rayleigh': H_Rayleigh})
        scio.savemat(os.path.join(path, "data\\H" + '_L_' + str(self.L) + '_setup_' + '.mat'), {'H': H})
        scio.savemat(os.path.join(path, "data\\HMean" + '_L_' + str(self.L) + '_setup_' + '.mat'), {'HMean': HMean_new})
        scio.savemat(os.path.join(path, "data\\HMeanx" + '_L_' + str(self.L) + '_setup_'  + '.mat'), {'HMeanx': HMeanx})
        scio.savemat(os.path.join(path, "data\\R" + '_L_' + str(self.L) + '_setup_' + '.mat'), {'R': R_new})
        scio.savemat(os.path.join(path, "data\\pilotIndex" + '_L_' + str(self.L) + '_setup_' + '.mat'), {'pilotIndex': pilotIndex})

        return H, R_new, pilotIndex

  def Caculate_rate(self, order, p):
    
        term_1 = np.zeros((self.L, self.L, self.K, self.J), dtype=np.complex128)
        term_2 = np.zeros((self.L, self.L, self.K, self.J), dtype=np.complex128)
        term_3 = np.zeros((self.L, self.L, self.K, self.J), dtype=np.complex128)
        term_4 = np.zeros((self.L, self.L, self.K, self.J), dtype=np.complex128)
        term_5 = np.zeros((self.L, self.L, self.K, self.J), dtype=np.complex128)

        denominator = np.zeros((self.L, self.L, self.K, self.J), dtype=np.complex128)
        SINR = np.zeros((self.K, self.J), dtype=np.complex128)

        # path = r'D:\CF_RS_HI_code\src'

        A_data = scio.loadmat(
            os.path.join(
                path,
                'data/A' + '_L_' + str(self.L) + '_setup_' + '.mat'
            )
        )

        Y_data = scio.loadmat(
            os.path.join(
                path,
                'data/Y' + '_L_' + str(self.L) + '_setup_' + '.mat'
            )
        )

        bk_data = scio.loadmat(
            os.path.join(
                path,
                'data/b_k' + '_L_' + str(self.L) + '_setup_' + '.mat'
            )
        )

        c_data = scio.loadmat(
            os.path.join(
                path,
                'data/c' + '_L_' + str(self.L) + '_setup_' + '.mat'
            )
        )

        A = A_data['A']
        Y = Y_data['Y']
        b_k = bk_data['b_k']
        c = c_data['c']

        B_b = np.zeros(
            (self.L, self.L, self.K, self.J),
            dtype=np.complex128
        )

        C_c = np.zeros(
            (self.L, self.L, self.K, self.K, self.J),
            dtype=np.complex128
        )

        for k in range(self.K):
            for j in range(self.J):
                b_col = b_k[k, :, j].reshape(self.L, 1)
                B_b[:, :, k, j] = b_col @ b_col.conj().T

        for k in range(self.K):
            for i in range(self.K):
                for j in range(self.J):
                    c_col = c[:, k, i, j].reshape(self.L, 1)
                    C_c[:, :, k, i, j] = c_col @ c_col.conj().T

        # =========================================================
        # term_1:
        # kappa * sum_{order(i,j)>=order(k,s)} p(i,j)*Y(:,:,k,i,j)
        # =========================================================
        for k in range(self.K):
            for s in range(self.J):
                for i in range(self.K):
                    for j in range(self.J):
                        if order[i, j] >= order[k, s]:
                            term_1[:, :, k, s] += (
                                self.kappa
                                * p[i, j]
                                * Y[:, :, k, i, j]
                            )

        # =========================================================
        # term_2:
        # kappa * sum_{order(k,j)>order(k,s)} p(k,j)*b_kj*b_kj^H
        # =========================================================
        for k in range(self.K):
            for s in range(self.J):
                for j in range(self.J):
                    if order[k, j] > order[k, s]:
                        term_2[:, :, k, s] += (
                            self.kappa
                            * p[k, j]
                            * B_b[:, :, k, j]
                        )

        # =========================================================
        # term_3:
        # kappa * sum_{order(i,j)>order(k,s)}
        # p(i,j)*c(:,k,i,j)*c(:,k,i,j)^H
        # =========================================================
        for k in range(self.K):
            for s in range(self.J):
                for i in range(self.K):
                    for j in range(self.J):
                        if order[i, j] > order[k, s]:
                            term_3[:, :, k, s] += (
                                self.kappa
                                * p[i, j]
                                * C_c[:, :, k, i, j]
                            )

        # =========================================================
        # term_4:
        # (1-kappa) * sum_all p(i,j)*(Y(:,:,k,i,j) + c*c^H)
        #
        # =========================================================
        for k in range(self.K):
            for s in range(self.J):
                for i in range(self.K):
                    for j in range(self.J):
                        term_4[:, :, k, s] += (
                            (1 - self.kappa)
                            * p[i, j]
                            * (
                                Y[:, :, k, i, j]
                                + C_c[:, :, k, i, j]
                            )
                        )

        # =========================================================
        # term_5:
        # A(:,:,k,s)
        # =========================================================
        for k in range(self.K):
            for s in range(self.J):
                term_5[:, :, k, s] = A[:, :, k, s]


        for k in range(self.K):
            for s in range(self.J):

                denominator[:, :, k, s] = (
                    term_1[:, :, k, s]
                    + term_2[:, :, k, s]
                    + term_3[:, :, k, s]
                    + term_4[:, :, k, s]
                    + term_5[:, :, k, s]
                )

                b_vec = b_k[k, :, s].reshape(self.L, 1)

                x = np.linalg.solve(
                    denominator[:, :, k, s],
                    b_vec
                )

                SINR[k, s] = (
                    self.kappa
                    * p[k, s]
                    * (b_vec.conj().T @ x)
                ).squeeze()

        return SINR


    def Generate_state(self, order, p, SINR, reward):
        # TODO 状态组成.
        # K*J + K + 1
        s_t = np.hstack([order.reshape(-1), p.reshape(-1), SINR.reshape(-1), reward])
        # s_t = np.hstack([reward, self.H_AP_UE.reshape(-1), self.H_AP_RIS.reshape(-1), self.H_RIS_UE.reshape(-1)])
        return s_t

    def Initial_para(self):
        '''
        返回值：
        '''
        # self.H, self.R, self.pilotIndex = self.Generate_H_set()

        #取随机动作
        order = np.zeros((self.K, self.J))
        p = np.ones((self.K, self.J))
        for i in range(self.K):
            for j in range(self.J):
                order[i, j] = i * self.J + j + 1
        # order_data = scio.loadmat('result_update/order.mat')
        # order = order_data['order']

        next_state, reward, SINR, done = self.Step(order, p)
        return next_state

    def Step(self, order, p):
       
        SINR = self.Caculate_rate(order, p)

        # 功率
        if np.all(np.sum(p, axis=1) <= self.Pmax):
            penalty = 1
        else:
            penalty = 0

        reward = np.sum(np.real(np.log2(1 + SINR))) * penalty

        next_state = self.Generate_state(order, p, SINR, reward)
        done = False

        return next_state, reward, SINR, done

    def reset(self):
        self.state = self.Initial_para()
        return self.state


if __name__ == '__main__':
    # p_n = -114.  # dBm
    # sigma2 = 1e-3 * pow(10., p_n / 10.)
    env = CF_RS_HI()
    env.Generate_H_set()
    env.functionChannelEstimates()
    # SE = env. Caculate_rate()
    # print(SE)
