import numpy as np
import time


def update_location_gannet_a(x, mx):
    updated_x = x + np.random.random() * (x - mx)
    return updated_x


def update_location_gannet_b(x, mx):
    updated_x = mx + np.random.random() * (x - mx)
    return updated_x

# Gannet Optimization Algorithm (GOA)
def GOA(X, fitness_function, lb, ub, Tmax_iter):
    N, Dim = X.shape[0], X.shape[1]

    # Generate memory matrix MX
    MX = np.copy(X)

    # Calculate the fitness value of X
    fitness_X = np.array([fitness_function(x) for x in X])
    convergence = np.zeros((Tmax_iter, 1))

    ct = time.time()
    # Main optimization loop
    for t in range(Tmax_iter):
        # Determine whether to use Equation (7a) or Equation (7b)
        if np.random.random() > 0.5:
            for i in range(N):
                q = np.random.random()
                if q >= 0.5:
                    # Update the location of Gannet using Equation (7a)
                    X[i, :] = update_location_gannet_a(X[i, :], MX[i, :])
                else:
                    # Update the location of Gannet using Equation (7b)
                    X[i, :] = update_location_gannet_b(X[i, :], MX[i, :])
        else:
            for i in range(N):
                c = np.random.random()
                if c >= 0.2:
                    # Update the location of Gannet using Equation (17a)
                    X[i, :] = update_location_gannet_a(X[i, :], MX[i, :])
                else:
                    # Update the location of Gannet using Equation (17b)
                    X[i, :] = update_location_gannet_b(X[i, :], MX[i, :])

        # Calculate the fitness value of X
        fitness_X = np.array([fitness_function(x) for x in X])

        # Update MX if the value of MXi is better than the value of Xi
        better_indices = np.where(fitness_X > fitness_function(MX))
        MX[better_indices] = X[better_indices]

        convergence[t, :] = np.min(X)
    # Find the best solution and its fitness value
    best_index = np.argmin(fitness_X)
    best_solution = X[best_index]
    best_fitness = fitness_X[best_index]

    ct += time.time()
    return best_fitness, convergence, best_solution, ct
