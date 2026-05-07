import numpy as np
from scipy.linalg import sqrtm, toeplitz
from scipy import integrate
import scipy
from scipy.linalg import sqrtm
from scipy.stats import norm
from src.config import *
import scipy.io as scio
# from tt import *
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

SSE_sum = np.zeros((50, 1), dtype=np.complex128)


def Cauculate_SSE(p, ak, b_k, Y, c, order):
    indices = np.argwhere(order == 1)
    Ind = indices

    for k in range(K):
        for s in range(J):
            term_DS[k, s] = kappa * p[k, s] * np.abs(
                ak[k, :, s].conj() @ (b_k[k, :, s].reshape(L, 1))) ** 2

    for ii in range(Ind.shape[0]):
        k = Ind[ii, 0]
        s = Ind[ii, 1]

        term1 = 0
        for i in range(K):
            for j in range(J):
                if order[i, j] >= order[k, s]:
                    term1 = term1 + p[i, j] * ak[k, :, s].conj() @ Y[:, :, k, i, j] @ ak[k, :, s].T
        term_1[k, s] = term1

        term2 = 0
        for j in range(J):
            if order[k, j] >= order[k, s] and j != s:
                term2 = term2 + kappa * p[k, j] * np.abs(
                    ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2

        term_2[k, s] = term2

        term3 = 0
        for i in range(K):
            for j in range(J):
                if order[i, j] >= order[k, s] and i != k:
                    term3 = term3 + p[i, j] * np.abs(
                        ak[k, :, s].conj() @ (c[:, k, i, j].reshape(L, 1))) ** 2
        term_3[k, s] = term3

        term4 = 0
        for j in range(J):
            if order[k, j] >= order[k, s]:
                term4 = term4 + (1 - kappa) * p[k, j] * np.abs(
                    ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
        term_4[k, s] = term4

    for k in range(K):
        for j in range(J):
            term_5[k, j] = ak[k, :, j].conj() @ A[:, :, k, j] @ ak[k, :, j].T

    for k in range(K):
        for s in range(J):
            SINR1[k, s] = term_DS[k, s] / (term_1[k, s] + term_2[k, s] + term_3[k, s] + term_4[k, s] + term_5[k, s])

    SE = np.real(np.log2(1 + SINR1))

    indices = np.argwhere(order == 0)
    Ind = indices

    for ii in range(Ind.shape[0]):
        k = Ind[ii, 0]
        s = Ind[ii, 1]

        SE[k, s] = 0

    return SE

print("实验开始时间:", time.asctime(time.localtime(time.time())))
start_time = time.time()
for ss in range(1):
    # path = r'E:\HI_DRL2\src'

    A_data = scio.loadmat(os.path.join(path, 'data/A_L_10_setup_'+ '.mat'))
    Y_data = scio.loadmat(os.path.join(path, 'data/Y_L_10_setup_' + '.mat'))
    bk_data = scio.loadmat(os.path.join(path, 'data/b_k_L_10_setup_' + '.mat'))
    c_data = scio.loadmat(os.path.join(path, 'data/c_L_10_setup_'  + '.mat'))
    R_data = scio.loadmat(os.path.join(path, 'data/R_L_10_setup_'  + '.mat'))

    A = A_data['A']
    Y = Y_data['Y']
    b_k = bk_data['b_k']
    c = c_data['c']
    R = R_data['R']

    ak = np.ones((K, L, J), dtype=np.complex128)
    term_1 = np.zeros((K, J), dtype=np.complex128)
    term_2 = np.zeros((K, J), dtype=np.complex128)
    term_3 = np.zeros((K, J), dtype=np.complex128)
    term_4 = np.zeros((K, J), dtype=np.complex128)
    term_5 = np.zeros((K, J), dtype=np.complex128)
    SINR1 = np.ones((K, J), dtype=np.complex128)
    term_DS = np.ones((K, J), dtype=np.complex128)
    order = np.ones((K, J))
    p = Pmax/J * np.ones((K, J), dtype=np.complex128)
    denominator = np.zeros((K, J), dtype=np.complex128)
    B = np.zeros((L, L, K, J), dtype=np.complex128)
    denominator1 = np.zeros((K, J), dtype=np.complex128)

    SSE = np.zeros((K, J))
    order_record = np.zeros((K, J, K, J))
    order_max = np.zeros((K, J))

    SE = 0
    for k in range(K):
        for s in range(J):
            order_ori = order
            SSE = np.zeros((K, J))
            order_record = np.zeros((K, J, K, J))
            order_max = np.zeros((K, J))

            for k in range(K):
                for s in range(J):
                    order = np.ones((K, J))
                    t = 1
                    SEo = Cauculate_SSE(p, ak, b_k, Y, c, order)
                    SSE[k, s] = SSE[k, s] + SEo[k, s]
                    order_record[k, s, k, s] = t

                    while not (np.array_equal(order, np.zeros((K, J)))) or np.array_equal(order, np.ones((K, J))):
                        order[k, s] = 0
                        SEo = Cauculate_SSE(p, ak, b_k, Y, c, order)
                        SE_max = SEo[:, s].max()
                        ind = np.argmax(SEo[:, s])
                        m = np.unravel_index(ind, SEo[:, s].shape)

                        order[m, s] = 0
                        t += 1
                        order_record[k, s, m, s] = t

                        if t == K:
                            s = 1-s
                        SSE[k, s] += SE_max

            SSE_max = SSE.max()
            ind = np.argmax(SSE)
            a, b = np.unravel_index(ind, SSE.shape)
            order_max = order_record[a, b]

            if SSE_max > SE:
                order = order_max
            else:
                order = order_ori

    SSE_max = SSE.max()
    ind = np.argmax(SSE)
    a, b = np.unravel_index(ind, SSE.shape)
    order_max = order_record[a, b]
    order = order_max

    for k in range(K):
        for s in range(J):
            ak1 = np.zeros((L, L), dtype=np.complex128)
            ak2 = np.zeros((L, L), dtype=np.complex128)
            ak3 = np.zeros((L, L), dtype=np.complex128)
            ak4 = np.zeros((L, L), dtype=np.complex128)

            for j in range(J):
                if order[k, j] > order[k, s]:
                    ak1 = ak1 + kappa * p[k, j] * (b_k[k, :, j].reshape(L, 1)) @ (b_k[k, :, j].reshape(1, L).conj())
                if order[k, j] >= order[k, s]:
                    ak2 = ak2 + (1 - kappa) * p[k, j] * b_k[k, :, j].reshape(L, 1) @ (b_k[k, :, j].reshape(1, L).conj())

            for i in range(K):
                for j in range(J):
                    if order[i, j] >= order[k, s]:
                        ak4 = ak4 + p[i, j] * Y[:, :, k, i, j]
                    if order[i, j] > order[k, s]:
                        ak3 = ak3 + p[i, j] * np.outer(c[:, k, i, j], c[:, k, i, j].conj())

            ak5 = A[:, :, k, s]
            B[:, :, k, s] = ak1 + ak2 + ak3 + ak4 + ak5
            ak[k, :, s] = np.linalg.inv((ak1 + ak2 + ak3 + ak4 + ak5)) @ (b_k[k, :, s].reshape(L, 1)).squeeze()

    def update_p(ak, b_k, Y, y, c, rho, order):
        for m in range(K):
            for n in range(J):
                de1 = 0
                de2 = 0
                de3 = 0
                for k in range(K):
                    for s in range(J):
                        if order[m, n] >= order[k, s] and k == m:
                            de1 = de1 + y[m, s] ** 2 * np.abs(ak[m, :, s].conj() @ (b_k[m, :, s].reshape(L, 1))) ** 2
                        if order[m, n] >= order[k, s]:
                            de2 = de2 + y[k, s] ** 2 * ak[k, :, s].conj() @ Y[:, :, k, m, s] @ ak[k, :, s].T
                        if order[m, n] > order[k, s]:
                            de3 = de3 + y[k, s] ** 2 * np.abs(ak[k, :, s].conj() @ (c[:, k, m, s].reshape(L, 1))) ** 2

                num = y[m, n] ** 2 * (1 + v[m, n]) * kappa * np.abs(ak[m, :, n].conj() @ (b_k[m, :, n].reshape(L, 1))) ** 2
                p[m, n] = (num / (de1 + de2 + de3 + rho[m]) ** 2).squeeze()
        return p

    for k in range(K):
        for s in range(J):
            term_DS[k, s] = kappa * p[k, s] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, s].reshape(L, 1))) ** 2

    for k in range(K):
        for s in range(J):
            term1 = 0
            for i in range(K):
                for j in range(J):
                    if order[i, j] >= order[k, s]:
                        term1 = term1 + p[i, j] * ak[k, :, s].conj() @ Y[:, :, k, i, j] @ ak[k, :, s].T
            term_1[k, s] = term1

    for k in range(K):
        for s in range(J):
            term2 = 0
            for j in range(J):
                if order[k, j] > order[k, s]:
                    term2 = term2 + kappa * p[k, j] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
            term_2[k, s] = term2

    for k in range(K):
        for s in range(J):
            term3 = 0
            for i in range(K):
                for j in range(J):
                    if order[i, j] > order[k, s]:
                       term3 = term3 + p[i, j] * np.abs(ak[k, :, s].conj() @ (c[:, k, i, j].reshape(L, 1))) ** 2
            term_3[k, s] = term3

    for k in range(K):
        for j in range(J):
            term_5[k, j] = ak[k, :, j].conj() @ A[:, :, k, j] @ ak[k, :, j].T

    for k in range(K):
        for s in range(J):
            term4 = 0
            for j in range(J):
                if order[k, j] >= order[k, s]:
                    term4 = term4 + (1 - kappa) * p[k, j] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
            term_4[k, s] = term4

    for k in range(K):
        for s in range(J):
            SINR1[k, s] = term_DS[k, s] / (term_1[k, s] + term_2[k, s] + term_3[k, s] + term_4[k, s] + term_5[k, s])
            denominator1[k, s] = term_DS[k, s] + term_1[k, s] + term_2[k, s] + term_3[k, s] + term_4[k, s] + term_5[k, s]

    SE1 = np.real(np.sum(np.log2(1 + SINR1), axis=1))
    print(np.sum(SE1))

    v = np.zeros((K, J), dtype=np.complex128)
    y = np.zeros((K, J), dtype=np.complex128)
    iterations = 50
    p = Pmax/J * np.ones((K, J), dtype=np.complex128)
    rho = np.zeros((K, 1), dtype=np.complex128)

    for ii in range(iterations):
        f1 = np.sum(np.log2(1 + v) - v + ((1 + v) * term_DS) / denominator1)
        L1 = 0
        for k in range(K):
            L1 = L1 + rho[k] * (Pmax - np.sum(p[k, :]))
        g = f1 + L1
        v = SINR1

        f2 = np.sum(np.log2(1 + v) - v + ((1 + v) * term_DS) / denominator1)
        L2 = 0
        for k in range(K):
            L2 = L2 + rho[k] * (Pmax - np.sum(p[k, :]))
        g2 = f2 + L2

        for k in range(K):
            for s in range(J):
                term_DS[k, s] = kappa * p[k, s] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, s].reshape(L, 1))) ** 2

        for k in range(K):
            for s in range(J):
                term1 = 0
                for i in range(K):
                    for j in range(J):
                        if order[i, j] >= order[k, s]:
                            term1 = term1 + p[i, j] * ak[k, :, s].conj() @ Y[:, :, k, i, j] @ ak[k, :, s].T
                term_1[k, s] = term1

        for k in range(K):
            for s in range(J):
                term2 = 0
                for j in range(J):
                    if order[k, j] > order[k, s]:
                        term2 = term2 + kappa * p[k, j] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
                term_2[k, s] = term2

        for k in range(K):
            for s in range(J):
                term3 = 0
                for i in range(K):
                    for j in range(J):
                        if order[i, j] > order[k, s]:
                            term3 = term3 + p[i, j] * np.abs(ak[k, :, s].conj() @ (c[:, k, i, j].reshape(L, 1))) ** 2
                term_3[k, s] = term3

        for k in range(K):
            for j in range(J):
                term_5[k, j] = ak[k, :, j].conj() @ A[:, :, k, j] @ ak[k, :, j].T

        for k in range(K):
            for s in range(J):
                term4 = 0
                for j in range(J):
                    if order[k, j] >= order[k, s]:
                        term4 = term4 + (1 - kappa) * p[k, j] * np.abs(
                            ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
                term_4[k, s] = term4

        for k in range(K):
            for s in range(J):
                denominator1[k, s] = term_DS[k, s] + term_1[k, s] + term_2[k, s] + term_3[k, s] + term_4[k, s] + term_5[k, s]
                y[k, s] = np.sqrt((1 + v[k, s]) * term_DS[k, s]) / denominator1[k, s]

        fq = 0
        for k in range(K):
            for s in range(J):
                fq += 2 * y[k, s] * np.sqrt((1 + v[k, s]) * term_DS[k, s]) - y[k, s] ** 2 * denominator1[k, s]

        f3 = np.sum(np.log2(1 + v) - v) + fq
        L3 = 0
        for k in range(K):
            L3 = L3 + rho[k] * (Pmax - np.sum(p[k, :]))
        g3 = f3 + L3
        p = update_p(ak, b_k, Y, y, c, rho, order)

        for m in range(K):
            low, high = 0.0, 5
            eps = 1e-10
            while (high - low) > eps:
                rho_mid = (low + high) / 2
                rho[m] = rho_mid
                p = update_p(ak, b_k, Y, y, c, rho, order)
                total_p = np.sum(p[m, :])
                if total_p > Pmax:
                    low = rho_mid
                else:
                    high = rho_mid
            rho[m] = low

        p = update_p(ak, b_k, Y, y, c, rho, order)

        for k in range(K):
            for s in range(J):
                term_DS[k, s] = kappa * p[k, s] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, s].reshape(L, 1))) ** 2

        for k in range(K):
            for s in range(J):
                term1 = 0
                for i in range(K):
                    for j in range(J):
                        if order[i, j] >= order[k, s]:
                            term1 = term1 + p[i, j] * ak[k, :, s].conj() @ Y[:, :, k, i, j] @ ak[k, :, s].T
                term_1[k, s] = term1

        for k in range(K):
            for s in range(J):
                term2 = 0
                for j in range(J):
                    if order[k, j] > order[k, s]:
                        term2 = term2 + kappa * p[k, j] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
                term_2[k, s] = term2

        for k in range(K):
            for s in range(J):
                term3 = 0
                for i in range(K):
                    for j in range(J):
                        if order[i, j] > order[k, s]:
                            term3 = term3 + p[i, j] * np.abs(ak[k, :, s].conj() @ (c[:, k, i, j].reshape(L, 1))) ** 2
                term_3[k, s] = term3

        for k in range(K):
            for j in range(J):
                term_5[k, j] = ak[k, :, j].conj() @ A[:, :, k, j] @ ak[k, :, j].T

        for k in range(K):
            for s in range(J):
                term4 = 0
                for j in range(J):
                    if order[k, j] >= order[k, s]:
                        term4 = term4 + (1 - kappa) * p[k, j] * np.abs(
                            ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
                term_4[k, s] = term4

        fq2 = 0
        for k in range(K):
            for s in range(J):
                denominator1[k, s] = term_DS[k, s] + term_1[k, s] + term_2[k, s] + term_3[k, s] + term_4[k, s] + term_5[k, s]
                fq2 += 2 * y[k, s] * np.sqrt((1 + v[k, s]) * term_DS[k, s]) - y[k, s] ** 2 * denominator1[k, s]

        f4 = np.sum(np.log2(1 + v) - v) + fq2
        L4 = 0
        for k in range(K):
            L4 = L4 + rho[k] * (Pmax - np.sum(p[k, :]))
        g4 = f4 + L4

        for k in range(K):
            for s in range(J):
                ak1 = np.zeros((L, L), dtype=np.complex128)
                ak2 = np.zeros((L, L), dtype=np.complex128)
                ak3 = np.zeros((L, L), dtype=np.complex128)
                ak4 = np.zeros((L, L), dtype=np.complex128)

                for j in range(J):
                    if order[k, j] > order[k, s]:
                        ak1 = ak1 + kappa * p[k, j] * (b_k[k, :, j].reshape(L, 1)) @ (b_k[k, :, j].reshape(1, L).conj())
                    if order[k, j] >= order[k, s]:
                        ak2 = ak2 + (1 - kappa) * p[k, j] * b_k[k, :, j].reshape(L, 1) @ (b_k[k, :, j].reshape(1, L).conj())

                for i in range(K):
                    for j in range(J):
                        if order[i, j] >= order[k, s]:
                            ak4 = ak4 + p[i, j] * Y[:, :, k, i, j]
                        if order[i, j] > order[k, s]:
                            ak3 = ak3 + p[i, j] * np.outer(c[:, k, i, j], c[:, k, i, j].conj())

                ak5 = A[:, :, k, s]
                B[:, :, k, s] = ak1 + ak2 + ak3 + ak4 + ak5
                ak[k, :, s] = np.linalg.inv((ak1 + ak2 + ak3 + ak4 + ak5)) @ (b_k[k, :, s].reshape(L, 1)).squeeze()

        for k in range(K):
            for s in range(J):
                term_DS[k, s] = kappa * p[k, s] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, s].reshape(L, 1))) ** 2

        for k in range(K):
            for s in range(J):
                term1 = 0
                for i in range(K):
                    for j in range(J):
                        if order[i, j] >= order[k, s]:
                            term1 = term1 + p[i, j] * ak[k, :, s].conj() @ Y[:, :, k, i, j] @ ak[k, :, s].T
                term_1[k, s] = term1

        for k in range(K):
            for s in range(J):
                term2 = 0
                for j in range(J):
                    if order[k, j] > order[k, s]:
                        term2 = term2 + kappa * p[k, j] * np.abs(ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
                term_2[k, s] = term2

        for k in range(K):
            for s in range(J):
                term3 = 0
                for i in range(K):
                    for j in range(J):
                        if order[i, j] > order[k, s]:
                            term3 = term3 + p[i, j] * np.abs(ak[k, :, s].conj() @ (c[:, k, i, j].reshape(L, 1))) ** 2
                term_3[k, s] = term3

        for k in range(K):
            for j in range(J):
                term_5[k, j] = ak[k, :, j].conj() @ A[:, :, k, j] @ ak[k, :, j].T

        for k in range(K):
            for s in range(J):
                term4 = 0
                for j in range(J):
                    if order[k, j] >= order[k, s]:
                        term4 = term4 + (1 - kappa) * p[k, j] * np.abs(
                            ak[k, :, s].conj() @ (b_k[k, :, j].reshape(L, 1))) ** 2
                term_4[k, s] = term4

        for k in range(K):
            for s in range(J):
                denominator1[k, s] = term_DS[k, s] + term_1[k, s] + term_2[k, s] + term_3[k, s] + term_4[k, s] + term_5[k, s]

        SINR1 = term_DS / (term_1 + term_2 + term_3 + term_4 + term_5)
        SE = np.sum(np.log2(1 + SINR1))

    SSE_sum[ss] = SE
    print(SSE_sum[ss])

duration = time.time() - start_time
print(f"完成实验 - 耗时: {duration:.2f}s")
print("当前时间:", time.asctime(time.localtime(time.time())))