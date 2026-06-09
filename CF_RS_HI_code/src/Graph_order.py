import os
import time

import numpy as np
import scipy.io as scio
from src.config import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def calculate_terms(p, ak, b_k, Y, c, A, order):

    term_DS = np.zeros((K, J), dtype=np.complex128)
    term_1 = np.zeros((K, J), dtype=np.complex128)
    term_2 = np.zeros((K, J), dtype=np.complex128)
    term_3 = np.zeros((K, J), dtype=np.complex128)
    term_4 = np.zeros((K, J), dtype=np.complex128)
    term_5 = np.zeros((K, J), dtype=np.complex128)

    for k in range(K):
        for s in range(J):
            a_ks = ak[k, :, s]

            term_DS[k, s] = kappa * p[k, s] * np.abs(
                a_ks.conj().T @ b_k[k, :, s]
            ) ** 2

            temp1 = 0
            for i in range(K):
                for j in range(J):
                    if order[i, j] >= order[k, s]:
                        temp1 += p[i, j] * (
                            a_ks.conj().T @ Y[:, :, k, i, j] @ a_ks
                        )
            term_1[k, s] = temp1

            temp2 = 0
            for j in range(J):
                if order[k, j] > order[k, s]:
                    temp2 += kappa * p[k, j] * np.abs(
                        a_ks.conj().T @ b_k[k, :, j]
                    ) ** 2
            term_2[k, s] = temp2

            temp3 = 0
            for i in range(K):
                for j in range(J):
                    if order[i, j] > order[k, s]:
                        temp3 += p[i, j] * np.abs(
                            a_ks.conj().T @ c[:, k, i, j]
                        ) ** 2
            term_3[k, s] = temp3

            temp4 = 0
            for j in range(J):
                if order[k, j] >= order[k, s]:
                    temp4 += (1 - kappa) * p[k, j] * np.abs(
                        a_ks.conj().T @ b_k[k, :, j]
                    ) ** 2
            term_4[k, s] = temp4

            term_5[k, s] = a_ks.conj().T @ A[:, :, k, s] @ a_ks

    denominator = term_1 + term_2 + term_3 + term_4 + term_5

    SINR1 = np.divide(
        term_DS,
        denominator,
        out=np.zeros_like(term_DS, dtype=np.complex128),
        where=np.abs(denominator) > 0,
    )

    SE_matrix = np.real(np.log2(1 + SINR1))
    SE_matrix[order == 0] = 0

    denominator1 = term_DS + denominator

    return {
        "term_DS": term_DS,
        "term_1": term_1,
        "term_2": term_2,
        "term_3": term_3,
        "term_4": term_4,
        "term_5": term_5,
        "denominator": denominator,
        "denominator1": denominator1,
        "SINR1": SINR1,
        "SE_matrix": SE_matrix,
    }


def Cauculate_SSE(p, ak, b_k, Y, c, A, order):

    return calculate_terms(
        p, ak, b_k, Y, c, A, order
    )["SE_matrix"]


def calculate_total_SE(p, ak, b_k, Y, c, A, order):

    terms = calculate_terms(p, ak, b_k, Y, c, A, order)
    return float(np.real(np.sum(np.log2(1 + terms["SINR1"]))))


# =========================================================
# Update ak
# =========================================================

def update_ak(p, ak, b_k, Y, c, A, order, B=None):

    if B is None:
        B = np.zeros((L, L, K, J), dtype=np.complex128)

    for k in range(K):
        for s in range(J):
            ak1 = np.zeros((L, L), dtype=np.complex128)
            ak2 = np.zeros((L, L), dtype=np.complex128)
            ak3 = np.zeros((L, L), dtype=np.complex128)
            ak4 = np.zeros((L, L), dtype=np.complex128)

            for j in range(J):
                b_col = b_k[k, :, j].reshape(L, 1)

                if order[k, j] > order[k, s]:
                    ak1 += kappa * p[k, j] * (b_col @ b_col.conj().T)

                if order[k, j] >= order[k, s]:
                    ak2 += (1 - kappa) * p[k, j] * (b_col @ b_col.conj().T)

            for i in range(K):
                for j in range(J):
                    if order[i, j] >= order[k, s]:
                        ak4 += p[i, j] * Y[:, :, k, i, j]

                    if order[i, j] > order[k, s]:
                        c_col = c[:, k, i, j].reshape(L, 1)
                        ak3 += p[i, j] * (c_col @ c_col.conj().T)

            ak5 = A[:, :, k, s]
            B[:, :, k, s] = ak1 + ak2 + ak3 + ak4 + ak5

            ak[k, :, s] = np.linalg.solve(B[:, :, k, s], b_k[k, :, s])

    return ak, B


def generate_order(p, ak, b_k, Y, c, A):

    final_order = np.zeros((K, J), dtype=np.float64)
    global_rank = 1

    for cur_s in range(J):

        best_path = None
        best_score = -np.inf

        for k_start in range(K):

            order_temp = np.ones((K, J), dtype=np.float64)
            selected = np.zeros(K, dtype=bool)

            path = []
            path_score = 0.0
            cur_k = k_start

            while len(path) < K:

                SEo = Cauculate_SSE(p, ak, b_k, Y, c, A, order_temp)
                SEo = np.real(SEo)

                path.append(cur_k)
                selected[cur_k] = True
                path_score += SEo[cur_k, cur_s]

                order_temp[cur_k, cur_s] = 0

                if len(path) == K:
                    break

                candidate_SE = SEo[:, cur_s].copy()
                candidate_SE[selected] = -np.inf

                cur_k = int(np.argmax(candidate_SE))

            if path_score > best_score:
                best_score = path_score
                best_path = path.copy()

        for k_user in best_path:
            final_order[k_user, cur_s] = global_rank
            global_rank += 1

    return final_order


def update_order(
    p, ak, b_k, Y, c, A, order, B=None, tol=1e-10
):

    order_old = order.copy()

    ak_old = ak.copy()
    B_old = None if B is None else B.copy()

    ak_old, B_old = update_ak(
        p, ak_old, b_k, Y, c, A, order_old, B_old
    )

    SE_old = calculate_total_SE(
        p, ak_old, b_k, Y, c, A, order_old
    )

    order_candidate = generate_order(
        p, ak_old, b_k, Y, c, A
    )

    ak_candidate = ak.copy()
    B_candidate = None if B is None else B.copy()

    ak_candidate, B_candidate = update_ak(
        p, ak_candidate, b_k, Y, c, A, order_candidate, B_candidate
    )

    SE_candidate = calculate_total_SE(
        p, ak_candidate, b_k, Y, c, A, order_candidate
    )

    if SE_candidate > SE_old + tol:
        return order_candidate, ak_candidate, B_candidate, SE_candidate, True

    return order_old, ak_old, B_old, SE_old, False


# =========================================================
# Update power
# =========================================================

def update_p(ak, b_k, Y, y, c, rho, order, p, v):

    for m in range(K):
        for n in range(J):
            de1 = 0
            de2 = 0
            de3 = 0

            for k in range(K):
                for s in range(J):

                    if order[m, n] >= order[k, s] and k == m:
                        de1 += y[m, s] ** 2 * np.abs(
                            ak[m, :, s].conj().T @ b_k[m, :, s]
                        ) ** 2

                    if order[m, n] >= order[k, s]:
                        de2 += y[k, s] ** 2 * (
                            ak[k, :, s].conj().T
                            @ Y[:, :, k, m, s]
                            @ ak[k, :, s]
                        )

                    if order[m, n] > order[k, s]:
                        de3 += y[k, s] ** 2 * np.abs(
                            ak[k, :, s].conj().T @ c[:, k, m, s]
                        ) ** 2

            num = y[m, n] ** 2 * (1 + v[m, n]) * kappa * np.abs(
                ak[m, :, n].conj().T @ b_k[m, :, n]
            ) ** 2

            den = de1 + de2 + de3 + rho[m]

            p[m, n] = (num / (den ** 2)).squeeze()

    return p


# =========================================================
# Main
# =========================================================

def main():

    SSE_sum = np.zeros((50, 1), dtype=np.complex128)

    for ss in range(1):

        path = r'D:\CF_RS_HI_code\src'

        A_data = scio.loadmat(os.path.join(path, "data/A_L_10_setup_.mat"))
        Y_data = scio.loadmat(os.path.join(path, "data/Y_L_10_setup_.mat"))
        bk_data = scio.loadmat(os.path.join(path, "data/b_k_L_10_setup_.mat"))
        c_data = scio.loadmat(os.path.join(path, "data/c_L_10_setup_.mat"))
        R_data = scio.loadmat(os.path.join(path, "data/R_L_10_setup_.mat"))

        A = A_data["A"]
        Y = Y_data["Y"]
        b_k = bk_data["b_k"]
        c = c_data["c"]
        R = R_data["R"]

        ak = np.ones((K, L, J), dtype=np.complex128)
        order = np.ones((K, J), dtype=np.float64)
        p = Pmax / J * np.ones((K, J), dtype=np.complex128)
        B = np.zeros((L, L, K, J), dtype=np.complex128)

        order = generate_order(
            p, ak, b_k, Y, c, A
        )

        ak, B = update_ak(
            p, ak, b_k, Y, c, A, order, B
        )

        terms = calculate_terms(
            p, ak, b_k, Y, c, A, order
        )

        term_DS = terms["term_DS"]
        term_1 = terms["term_1"]
        term_2 = terms["term_2"]
        term_3 = terms["term_3"]
        term_4 = terms["term_4"]
        term_5 = terms["term_5"]
        denominator1 = terms["denominator1"]
        SINR1 = terms["SINR1"]

        SE = calculate_total_SE(
            p, ak, b_k, Y, c, A, order
        )

        # =====================================================
        # Power iteration
        # =====================================================
        v = np.zeros((K, J), dtype=np.complex128)
        y = np.zeros((K, J), dtype=np.complex128)
        iterations = 50
        p = Pmax / J * np.ones((K, J), dtype=np.complex128)
        rho = np.zeros((K, 1), dtype=np.complex128)

        for ii in range(iterations):

            f1 = np.sum(
                np.log2(1 + v)
                - v
                + ((1 + v) * term_DS) / denominator1
            )

            L1 = 0
            for k in range(K):
                L1 += rho[k] * (Pmax - np.sum(p[k, :]))
            g = f1 + L1

            v = SINR1

            f2 = np.sum(
                np.log2(1 + v)
                - v
                + ((1 + v) * term_DS) / denominator1
            )

            L2 = 0
            for k in range(K):
                L2 += rho[k] * (Pmax - np.sum(p[k, :]))
            g2 = f2 + L2

            terms = calculate_terms(
                p, ak, b_k, Y, c, A, order
            )

            term_DS = terms["term_DS"]
            term_1 = terms["term_1"]
            term_2 = terms["term_2"]
            term_3 = terms["term_3"]
            term_4 = terms["term_4"]
            term_5 = terms["term_5"]
            denominator1 = terms["denominator1"]

            for k in range(K):
                for s in range(J):
                    y[k, s] = (
                        np.sqrt((1 + v[k, s]) * term_DS[k, s])
                        / denominator1[k, s]
                    )

            fq = 0
            for k in range(K):
                for s in range(J):
                    fq += (
                        2
                        * y[k, s]
                        * np.sqrt((1 + v[k, s]) * term_DS[k, s])
                        - y[k, s] ** 2 * denominator1[k, s]
                    )

            f3 = np.sum(np.log2(1 + v) - v) + fq

            L3 = 0
            for k in range(K):
                L3 += rho[k] * (Pmax - np.sum(p[k, :]))
            g3 = f3 + L3

            p = update_p(
                ak, b_k, Y, y, c, rho, order, p, v
            )

            for m in range(K):

                low, high = 0.0, 5.0
                eps = 1e-10

                while (high - low) > eps:

                    rho_mid = (low + high) / 2
                    rho[m] = rho_mid

                    p = update_p(
                        ak, b_k, Y, y, c, rho, order, p, v
                    )

                    total_p = np.real(np.sum(p[m, :]))

                    if total_p > Pmax:
                        low = rho_mid
                    else:
                        high = rho_mid

                rho[m] = low

            p = update_p(
                ak, b_k, Y, y, c, rho, order, p, v
            )

            terms = calculate_terms(
                p, ak, b_k, Y, c, A, order
            )

            term_DS = terms["term_DS"]
            term_1 = terms["term_1"]
            term_2 = terms["term_2"]
            term_3 = terms["term_3"]
            term_4 = terms["term_4"]
            term_5 = terms["term_5"]
            denominator1 = terms["denominator1"]

            fq2 = 0
            for k in range(K):
                for s in range(J):
                    fq2 += (
                        2
                        * y[k, s]
                        * np.sqrt((1 + v[k, s]) * term_DS[k, s])
                        - y[k, s] ** 2 * denominator1[k, s]
                    )

            f4 = np.sum(np.log2(1 + v) - v) + fq2

            L4 = 0
            for k in range(K):
                L4 += rho[k] * (Pmax - np.sum(p[k, :]))
            g4 = f4 + L4

            # =================================================
            # Update ak
            # =================================================
            ak, B = update_ak(
                p, ak, b_k, Y, c, A, order, B
            )

            terms = calculate_terms(
                p, ak, b_k, Y, c, A, order
            )

            term_DS = terms["term_DS"]
            term_1 = terms["term_1"]
            term_2 = terms["term_2"]
            term_3 = terms["term_3"]
            term_4 = terms["term_4"]
            term_5 = terms["term_5"]
            denominator1 = terms["denominator1"]
            SINR1 = terms["SINR1"]

            SE_before_order = calculate_total_SE(
                p, ak, b_k, Y, c, A, order
            )

            # =================================================
            # Layer-wise order update inside iteration
            # =================================================
            order_new, ak_new, B_new, SE_after_order, order_improved = (
                update_order(
                    p, ak, b_k, Y, c, A, order, B
                )
            )

            if order_improved:

                order = order_new.copy()
                ak = ak_new.copy()
                B = B_new.copy()
                SE = SE_after_order

                terms = calculate_terms(
                    p, ak, b_k, Y, c, A, order
                )

                term_DS = terms["term_DS"]
                term_1 = terms["term_1"]
                term_2 = terms["term_2"]
                term_3 = terms["term_3"]
                term_4 = terms["term_4"]
                term_5 = terms["term_5"]
                denominator1 = terms["denominator1"]
                SINR1 = terms["SINR1"]

            else:
                SE = SE_before_order

            print(
                f"Iter {ii + 1:03d}: "
                f"SE={SE:.6f}, "
                f"order_improved={order_improved}"
            )

        SSE_sum[ss] = SE


if __name__ == "__main__":
    main()
