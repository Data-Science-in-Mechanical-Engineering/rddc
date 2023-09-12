import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.ndimage import gaussian_filter

def hankel_matrix(data, start, rows, cols):
    """
    create a Hankel matrix of the form

    x_i ------- x_i+N-1
    |    ..        |
    |        ..    |
    x_i+L-1 --- x_i+N+L-2

    L = rows
    N = cols
    
    each sample stretches over dim[0], sampling stretches over dim[1]

    if individual samples of data are vectors (dim[0]>1),
    these are arranged as column vectors in the hankel matrix as well
    """
    assert(data.shape[1]>rows+cols-2-start) # we have enough samples
    n = data.shape[0]
    H = np.zeros((n*rows, cols))
    for i in range(rows):
        for j in range(cols):
            H[ n*i : n*(i+1), j] = data[:, start+i+j]

    return H

def spectral_radius(A):
    """
    calculate the maximum of the absolute value of the matrix eigenvalues
    """
    return np.max(np.abs(np.linalg.eigvals(A)))

def check_controllability(A, B, tol=None):
    """
    check controllability of a system with system matrix A and input matrix B
    """
    n = A.shape[0]
    if n>1:
        Q_C = np.hstack((B, [A**i@B for i in range(1,n)][0])) # controllability matrix
    else:
        Q_C = B
    #print("SVD of the controllability matrix: {0}".format(np.linalg.svd(Q_C)[1]))
    return np.linalg.matrix_rank(Q_C, tol=tol) == n

def check_stability(A, eps=1e-6):
    """
    check if spectral radius is less than 1-eps
    """
    return spectral_radius(A)<1-eps

def check_persistent_excitation_willems(states, inputs, tol=None):
    matrix = np.vstack((states, inputs))
    #print("SVD of the trajectory: {0}".format(np.linalg.svd(matrix)[1]))
    return np.linalg.matrix_rank(matrix, tol=tol) == matrix.shape[0]

def check_persistent_excitation_berberich(noisy_system, inputs, noise, tol=None):
    A = noisy_system['A']
    B = noisy_system['B']
    B_w = noisy_system['B_w']
    controllable = check_controllability(A, np.hstack((B, B_w)), tol=tol)
    full_noise_rank = np.linalg.matrix_rank(np.vstack((noise, inputs)), tol=tol)
    return controllable and full_noise_rank

def check_sys_is_in_sigma(noisy_system, noise, trajectory):
    """
    checks if a given system with A, B, B_w is inside the uncertain-loop 
    parameterization \Sigma_{X, U} (acc. to Berberich19) for a given noise instance
    """
    X_all = trajectory['state']
    U0 = trajectory['input']
    T = U0.shape[1]
    X0 = hankel_matrix(X_all, 0, 1, T) # X acc. to Berberich
    X1 = hankel_matrix(X_all, 1, 1, T) # X_plus acc. to Berberich
    A = noisy_system['A']
    B = noisy_system['B']
    B_w = noisy_system['B_w']
    W = noise #noise instance
    return X1 == A @ X0 + B @ U0 + B_w @ W

def recover_noise(noisy_system, trajectory):
    """
    """
    U0 = trajectory['U0']
    X0 = trajectory['X0']
    X1 = trajectory['X1']
    T = U0.shape[1]
    A = np.atleast_2d(noisy_system['A'])
    B = np.atleast_2d(noisy_system['B'])
    B_w = np.atleast_2d(noisy_system['B_w'])
    m_w = B_w.shape[1]
    assert B_w == np.eye(m_w), "Noise recovery only works for an identity matrix B_w"
    W = X1 - A @ X0 - B @ U0
    return W

def check_assumption_3(W, noiseInfo, tol=0):
    """
    Check assumption 3 for given noise instance as in Berberich et al. (2019)
    """
    assumedBound = noiseInfo['assumedBound']
    m_w = W.shape[0]
    T = W.shape[1]
    Q_w = -np.eye(m_w)
    S_w = np.zeros((m_w, T))
    R_w = assumedBound**2 * np.eye(T) * T

    P_w = np.block([[Q_w, S_w], 
                    [S_w.T, R_w]])
    I = np.eye(T)
    quadraticTerm = np.vstack([W, I]).T @ P_w @ np.vstack([W, I])
    return np.all(np.linalg.eigvals(quadraticTerm)>=tol)

def check_assumption_1(W, noiseInfo, tol=0):
    """
    Check assumption 1 for given noise instance as in van Waarde et al. (2020)
    """
    assumedBound = noiseInfo['assumedBound']
    m_w = W.shape[0]
    T = W.shape[1] # length of one trajectory
    Phi_11 = assumedBound**2 * np.eye(m_w) * T
    Phi_12 = np.zeros((m_w, T))
    Phi_22 = -np.eye(T)
    Phi = np.block([[Phi_11  , Phi_12],
                    [Phi_12.T, Phi_22]])
    I = np.eye(m_w)
    quadraticTerm = np.vstack([I, W.T]).T @ Phi @ np.vstack([I, W.T])
    return np.all(np.linalg.eigvals(quadraticTerm)>=tol)

def check_gen_slater_condition(U0, X0, X1, Phi):
    """
    Check generalized slater condition as in van Waarde et al. (2020)
    """
    n = X0.shape[0]
    m = U0.shape[0]
    trajM_cut = np.block([  [np.eye(n)      ,  X1],
                            [np.zeros((n,n)), -X0],
                            [np.zeros((m,n)), -U0]])
    try:
        slater = trajM_cut @ Phi @ trajM_cut.T
        eigs = np.linalg.eigvals(slater)
        # if np.linalg.cond(slater)>1e5:
            # print('Failed Slater due to condition number')
            # return False
        # print('Slater condition check -- eigenvalues of N:{0}'.format(eigs))
        return np.sum(eigs>0)>=n
    except np.linalg.LinAlgError: #ill-conditioned trajectories don't satisfy slater's condition
        print('Failed Slater due to numeric errors')
        return False

def plot_systems4synth(systems4synth, color='black', idx_to_plot=None, ax=None):
    """
    For a 1-D system, create a scatter plot of a set of systems
    by plotting all chosen B-coefficients over A-coefficients.
    """
    if ax is None:
        ax = plt.gca()
    if idx_to_plot is None:
        idx_to_plot = [i for i in range(len(systems4synth))]
    As = [systems4synth[idx][0] for idx in idx_to_plot]
    Bs = [systems4synth[idx][1] for idx in idx_to_plot]
    plot = ax.scatter(As, Bs, marker='x', c=color, s=10, linewidth=0.5)
    return plot

def plot_stable_region(K, hatch='/', from_x=-2, to_x=3):
    x = np.array([[from_x, to_x]])
    y1 = 1/K - x/K
    y2 = -1/K - x/K
    plot = plt.fill_between(x[0], y1[0], y2[0], color='green', alpha=0.2, hatch=hatch)
    return plot


def draw_sigma_XU(trajectory, noiseInfo, noise_criterion=1, limA=[-0.5, 2.0], limB=[-0.5, 2.0], 
                    resolution=100, colormap="binary", hatch="X", ax=None):
    """
    For a 1-D system, draws the $\Sigma_{X,U}$ 
    set of all systems that are consistent with
    the provided trajectory and noise information
    """
    if ax is None:
        ax = plt.gca()
    start = perf_counter()
    assert noise_criterion in [1,3], "noise_criterion must be either 1 or 3" #1 - van Waarde, 3 - Berberich
    As = np.linspace(*limA, resolution)
    Bs = np.linspace(*limB, resolution)
    A_mesh, B_mesh = np.meshgrid(As, Bs)
    B_w = noiseInfo['B_w']
    in_sigma = np.zeros_like(A_mesh)
    for ia in range(len(As)):
        a = As[ia]
        for ib in range(len(Bs)):
            b = Bs[ib]
            noisy_system = {'A':a, 'B':b, 'B_w':B_w}
            W = recover_noise(noisy_system, trajectory)
            if noise_criterion==1:
                if check_assumption_1(W, noiseInfo):
                    in_sigma[ib, ia] = 1.0
            elif noise_criterion==3:
                if check_assumption_3(W, noiseInfo):
                    in_sigma[ib, ia] = 1.0
    end = perf_counter()
    print(f"Evaluation time: {end - start:.6f} seconds")
    start = perf_counter()
    smoothing = 1.5  # Adjust this value for the desired smoothness
    smoothed_in_sigma = gaussian_filter(in_sigma.astype(float), sigma=smoothing)

    # plt.pcolormesh(A_mesh, B_mesh, in_sigma, cmap='binary', shading='nearest', vmin=0, vmax=1, alpha=0.2*in_sigma)
    plot, _ = ax.contourf(A_mesh, B_mesh, smoothed_in_sigma, levels=[0.5, 1], cmap=colormap, alpha=1.0, vmin=0, vmax=1, hatches=hatch).legend_elements()
    end = perf_counter()
    print(f"Plotting time: {end - start:.6f} seconds")

    return plot[0]