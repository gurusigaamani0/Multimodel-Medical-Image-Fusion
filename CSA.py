import time
from operator import xor
import numpy as np


def CSA(chameleonPositions, fobj, lb, ub, iteMax):
    N, dim = chameleonPositions.shape[0], chameleonPositions.shape[1]
    if ub.shape[2 - 1] == 1:
        ub = np.ones((1, dim)) * ub
        lb = np.ones((1, dim)) * lb
    cg_curve = np.zeros((1, iteMax))
    gPosition = np.zeros((10, 2))
    fmin0 = np.zeros((10, 1))
    # Evaluate the fitness of the initial population
    fit = np.zeros((iteMax))
    ct = time.time()
    for i in range(iteMax):
        fit[i] = fobj(chameleonPositions[i, :])
        # Initalize the parameters of CSA
        fitness = fit
        fmin0, index = np.amin(fitness), np.argmin(fitness)
        chameleonBestPosition = chameleonPositions
        gPosition = chameleonPositions[index, :]
        v = 0.1 * chameleonBestPosition
        v0 = 0.0 * v
        # Start CSA
        # Main parameters of CSA
        rho = 1.0
        gamma = 2.0
        alpha = 4.0
        beta = 3.0
        # Start CSA
        for t in range(iteMax):
            a = 2590 * (1 - np.exp(- np.log(t)))
            omega = (1 - (t / iteMax)) ** (rho * np.sqrt(t / iteMax))
            p1 = 2 * np.exp(- 2 * (t / iteMax) ** 2)
            p2 = 2 / (1 + np.exp((- t + iteMax / 2) / 100))
            mu = gamma * np.exp(- (alpha * t / iteMax) ** beta)
            # Update the position of CSA (Exploration)
            for i in range(N):
                if np.random.rand() >= 0.1:

                    chameleonPositions[i, :] = chameleonPositions[i, :] + p1 * (chameleonBestPosition[i, :]) - (
                    chameleonPositions[i, :]) * np.random.rand() + + p2 * (
                                                           gPosition - chameleonPositions[i, :]) * np.random.rand()
                else:
                    for j in range(dim):
                        chameleonPositions[i, j] = gPosition[j] + mu * (
                                (ub[i, j] - lb[i, j]) * np.random.rand() + lb[i, j]) * np.sign(
                            np.random.rand() - 0.5)
            # Rotation of the chameleons - Update the position of CSA (Exploitation)
            # Rotation 180 degrees in both direction or 180 in each direction
            # [chameleonPositions] = rotation(chameleonPositions, searchAgents, dim);
            #  # Chameleon velocity updates and find a food source
            for i in range(N):
                v[i, :] = omega * v[i, :] + p1 * (
                            chameleonBestPosition[i, :] - chameleonPositions[i, :]) * np.random.rand() + + p2 * (
                                  gPosition - chameleonPositions[i, :]) * np.random.rand()
                chameleonPositions[i, :] = chameleonPositions[i, :] + (v[i, :] ** 2 - v0[i, :] ** 2) / (2 * a)
            v0 = v
            for i in range(N):
                ub_ = np.sign(chameleonPositions[i, :] - ub) > 0
                lb_ = np.sign(chameleonPositions[i, :] - lb) < 0
                chameleonPositions[i, :] = (np.multiply(chameleonPositions[i, :],
                                                        (xor(lb_[i, :], ub_[i, :])))) + np.multiply(ub[i, :],
                                                                                                    ub_[i,
                                                                                                    :]) + np.multiply(
                    lb[i, :], lb_[i, :])
                fit[i] = fobj(chameleonPositions[i, :])
                if fit[i] < fitness[i]:
                    chameleonBestPosition[i, :] = chameleonPositions[i, :]
                    fitness[i] = fit[i]
            # Evaluate the new positions
            fmin, index = np.amin(fitness), np.argmin(fitness)
            # Updating gPosition and best fitness
            if fmin < fmin0:
                gPosition = chameleonBestPosition[index, :]
                fmin0 = fmin
            cg_curve[t] = fmin0
            g_best = max(fitness)
            fmin0 = fobj(g_best)
        ct = time.time() - ct
    return fmin0, gPosition, cg_curve, ct
