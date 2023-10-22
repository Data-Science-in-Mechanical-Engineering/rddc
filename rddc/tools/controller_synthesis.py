import numpy as np
import picos as pc
from rddc.tools import control_utils
from scipy.linalg import solve_discrete_are

def sysId_ls_lqr(trajectories, sysInfo, verbosity=1):
    """
    Performs Least Squares identification of the system based on given trajectories
    Synthesizes an LQR controller based on the identified system
    """
    m = sysInfo['m']
    n = sysInfo['n']
    Q = sysInfo['Q']
    R = sysInfo['R']
    U0 = np.hstack([trajectories[sysId]['U0'] for sysId in range(len(trajectories))])
    X0 = np.hstack([trajectories[sysId]['X0'] for sysId in range(len(trajectories))])
    X1 = np.hstack([trajectories[sysId]['X1'] for sysId in range(len(trajectories))])
    BA = np.linalg.lstsq(np.block([[U0],[X0]]).T, X1.T, rcond=None)[0].T
    B = BA[:, :m]
    A = BA[:, -n:]
    X = np.array(np.array(solve_discrete_are(A, B, Q, R)))
    K = - np.linalg.inv(R + B.T @ X @ B) @ (B.T @ X @ A)
    if verbosity>0:
        print("Least squares A is: \n{0}".format(np.array_str(A, precision=3, suppress_small=True)))
        print("Least squares B is: \n{0}".format(np.array_str(B, precision=3, suppress_small=True)))
        print('\nSpectral radius of the identified open loop: \n{}\n'.format(control_utils.spectral_radius(A)))
        if control_utils.check_controllability(A, B, tol=None):
            print('Identified system is controllable')
        else:
            print('Identified system is not controllable')
        print('\nOptimal syID LS LQR controller is found: \n{}\n'.format(K))
    return K

def robust_lqr_scenario(trajectories, noiseInfo, perfInfo, verbosity=1):
    """
    robust LQR controller stabilizing each one of given trajectories and optimizing performance for them.

    @input: trajectories: list of dictionaries containing state sequence under 'state' and input sequence under 'input'. 'state' must have one more entry than 'input'
    @input: noiseInfo: dictionary containing matrices B_w, Q_w, S_w, R_w describing noise parameters and influence on the system
    @input: perfInfo: dictionary containing matrices Q, S, R describing performance metrics

    @ouput: K: designed controller. Nan if optimizer didn't succeed.

    robustness to disturbances and stabilization is achieved via formulation acc. to:
    Berberich, Julian, Anne Koch, Carsten W. Scherer, und Frank Allgower. 2020.
    „Robust data-driven state-feedback design“. In 2020 American Control Conference (ACC),
    1532-38. Denver, CO, USA: IEEE. https://doi.org/10.23919/ACC45564.2020.9147320.

    robustness to system uncertainty is achieved via scenario approach:
    Calafiore, G.C., und M.C. Campi. 2006. „The Scenario Approach to Robust Control Design“.
    IEEE Transactions on Automatic Control 51 (5): 742-53. https://doi.org/10.1109/TAC.2006.875041.

    optimal peformance is achieved by adapting:
    De Persis, Claudio, und Pietro Tesi. 2020. „Formulas for Data-Driven Control: Stabilization, Optimality, and Robustness“.
    IEEE Transactions on Automatic Control 65 (3): 909-24. https://doi.org/10.1109/TAC.2019.2959924.
    """

    # Input parsing
    N = len(trajectories) # number of trajectories
    T = trajectories[0]['U0'].shape[1] # length of one trajectory
    n = trajectories[0]['X0'].shape[0] # number of states
    m = trajectories[0]['U0'].shape[0] # number of inputs

    B_w = noiseInfo['B_w']
    Q_w = noiseInfo['Q_w']
    S_w = noiseInfo['S_w']
    R_w = noiseInfo['R_w']

    Q = perfInfo['Q']
    S = perfInfo['S']
    R = perfInfo['R']

    m_w = B_w.shape[1]

    # The optimization problem (common variables)
    problem = pc.Problem()
    solved = False
    problem.options.solver = 'mosek'
    problem.set_option('mosek_params', {'MSK_IPAR_NUM_THREADS': 1})

    v_Y = pc.SymmetricVariable('Y', (n, n)) # = X0*Q, acc. to De Persis
    v_X = pc.SymmetricVariable('X', (m, m))
    v_U0M = pc.RealVariable('U0M', (m, n)) # product of U0 and M, which should be common for all systems
    v_Ms = [pc.RealVariable('M[{0}]'.format(trajId), (T, n)) for trajId in range(N)]

    problem.set_objective('min', pc.trace(Q*v_Y) + pc.trace(v_X))
    #problem.set_objective(None)

    perf00 = v_X
    perf01 = np.sqrt(R)  * v_U0M
    perf11 = v_Y
    perf = (
        (perf00     & perf01) //
        (perf01.T   & perf11)
    )

    problem.add_constraint(perf >> 0)
    problem.add_constraint(v_Y >> 0)
    problem.add_constraint(v_X >> 0)
    # Trajectory-dependent constraints for each trajectory
    for trajId in range(N):

        U0 = trajectories[trajId]['U0']
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']

        v_M = v_Ms[trajId]

        stab00 = -v_Y
        stab10 = -S_w*v_M
        stab20 = X1*v_M
        stab30 = v_M
        stab11 = Q_w
        stab21 = B_w
        stab31 = np.zeros((T, m_w))
        stab22 = -v_Y
        stab32 = np.zeros((T, n))
        stab33 = -np.linalg.inv(R_w)
        stab = (
            (stab00 & stab10.T & stab20.T & stab30.T) //
            (stab10 & stab11   & stab21.T & stab31.T) //
            (stab20 & stab21   & stab22   & stab32.T) //
            (stab30 & stab31   & stab32   & stab33)
        )

        problem.add_constraint(stab << 0)
        problem.add_constraint(X0 * v_M == v_Y)
        problem.add_constraint(U0 * v_M == v_U0M)

    solution = None
    try:
        solver_verbosity = max(verbosity-1, 0)
        solution = problem.solve(verbosity=solver_verbosity)
    except pc.modeling.problem.SolutionFailure:
        if verbosity>0:
            print("Solution Failure occured")
        # pass # solution failure is raised for infeasible problems
    except ValueError:
        if verbosity>0:
            print("Value Error occured, problem will count as not solved")
    except Exception as e: #ill-conditioned matrices, unexpected exceptions
        # if verbosity>0:
        print(e)
        print("That's a caught exception, continuing with the rest of the programm")
        pass

    #print(problem.status)
    if solution is not None:
        if solution.status=='primal feasible' and solution.primalStatus=='optimal':
            solved = True
        else:
            if verbosity>0:
                print(f"Problem unsolved. solution.status: {solution.status}. solution.primalStatus: {solution.primalStatus}")
    else:
        if verbosity>0:
            print("Problem could not be solved")
        pass

    K = np.nan
    if solved:
        U0M = np.atleast_2d(v_U0M.value)
        Y = np.atleast_2d(v_Y.value)
        K = U0M @ np.linalg.inv(Y)
        #print('\nOptimal controller is found: \n{}\n'.format(K))

    return K


def regularized_lqr_scenario(trajectories, noiseInfo, perfInfo, reg_factor=1e-1, verbosity=0):
    # Input parsing
    N = len(trajectories) # number of trajectories
    T = trajectories[0]['input'].shape[1] # length of one trajectory (TODO: should it be equal for all trajectories?)
    n = trajectories[0]['X0'].shape[0] # number of states
    m = trajectories[0]['U0'].shape[0] # number of inputs

    Q = perfInfo['Q']
    R = perfInfo['R']

    # The optimization problem
    problem = pc.Problem()
    solved = False
    problem.options.solver = 'mosek'
    problem.set_option('mosek_params', {'MSK_IPAR_NUM_THREADS': 1})

    # decision variables
    v_X     = pc.SymmetricVariable('X', (m, m))
    v_Ys    = [pc.RealVariable('Y[{0}]'.format(trajId), (T, n)) for trajId in range(N)]
    v_P     = pc.SymmetricVariable('P', (n, n))
    v_U0Y   = pc.RealVariable('U0Y', (m, n)) # product of U0 and Y, which should be common for all systems
    v_t     = pc.RealVariable('t')

    norms = list()

    perf00 = v_X
    perf01 = np.sqrt(R)  * v_U0Y
    perf11 = v_P
    perf = (
        (perf00     & perf01) //
        (perf01.T   & perf11)
    )
    problem.add_constraint(perf >> 0)

    problem.add_constraint(v_P - np.eye(n) >> 0)
    #problem.add_constraint(v_X >> 0)

    perf_opt = pc.trace(Q * v_P) + pc.trace(v_X)
    problem.add_constraint(v_t >= perf_opt)
    # Trajectory-dependent constraints for each trajectory
    for trajId in range(N):

        U0 = trajectories[trajId]['U0']
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']

        W0 = np.vstack((U0, X0))
        Pi = np.eye(T) - np.linalg.pinv(W0) @ W0
        v_Y = v_Ys[trajId]
        norms.append(pc.Norm(Pi * v_Y))
        #norms.append(pc.trace(v_Y * v_P * v_Y.T))

        stab00 = v_P - np.eye(n)
        stab01 = X1 * v_Y
        stab11 = v_P
        stab = (
            (stab00     & stab01 ) //
            (stab01.T   & stab11   )
        )


        problem.add_constraint(stab >> 0)
        problem.add_constraint(X0 * v_Y == v_P)
        problem.add_constraint(U0 * v_Y == v_U0Y)

    problem.set_objective('min', v_t + reg_factor * pc.sum(norms))

    solution = None
    try:
        solver_verbosity = max(verbosity-1, 0)
        solution = problem.solve(verbosity=solver_verbosity)
    except pc.modeling.problem.SolutionFailure:
        if verbosity>0:
            print("Solution Failure occured")
        # pass # solution failure is raised for infeasible problems
    except ValueError:
        if verbosity>0:
            print("Value Error occured, problem will count as not solved")
    except Exception as e: #ill-conditioned matrices, unexpected exceptions
        # if verbosity>0:
        print(e)
        print("That's a caught exception, continuing with the rest of the programm")
        pass

    #print(problem.status)
    if solution is not None:
        if solution.status=='primal feasible' and solution.primalStatus=='optimal':
            solved = True
        else:
            if verbosity>0:
                print(f"Problem unsolved. solution.status: {solution.status}. solution.primalStatus: {solution.primalStatus}")
    else:
        if verbosity>0:
            print("Problem could not be solved")
        pass

    K = np.nan
    if solved:
        U0Y = np.atleast_2d(v_U0Y.value)
        P = np.atleast_2d(v_P.value)
        K = U0Y @ np.linalg.inv(P)
        #print('\nOptimal controller is found: \n{}\n'.format(K))
    return K



def robust_scenario(trajectories, noiseInfo, perfInfo, verbosity=1):
    """
    robust controller stabilizing each one of given trajectories.

    @input: trajectories: list of dictionaries containing state sequence under 'state' and input sequence under 'input'. 'state' must have one more entry than 'input'
    @input: noiseInfo: dictionary containing matrices B_w, Q_w, S_w, R_w describing noise parameters and influence on the system

    @ouput: K: designed controller. Nan if optimizer didn't succeed.

    robustness to disturbances stabilization is achieved via formulation acc. to:
    Berberich, Julian, Anne Koch, Carsten W. Scherer, und Frank Allgower. 2020.
    „Robust data-driven state-feedback design“. In 2020 American Control Conference (ACC),
    1532-38. Denver, CO, USA: IEEE. https://doi.org/10.23919/ACC45564.2020.9147320.

    robustness to system uncertainty is achieved via scenario approach:
    Calafiore, G.C., und M.C. Campi. 2006. „The Scenario Approach to Robust Control Design“.
    IEEE Transactions on Automatic Control 51 (5): 742-53. https://doi.org/10.1109/TAC.2006.875041.
    """

    # Input parsing
    N = len(trajectories) # number of trajectories
    T = trajectories[0]['input'].shape[1] # length of one trajectory
    n = trajectories[0]['X0'].shape[0] # number of states
    m = trajectories[0]['U0'].shape[0] # number of inputs

    B_w = noiseInfo['B_w']
    Q_w = noiseInfo['Q_w']
    S_w = noiseInfo['S_w']
    R_w = noiseInfo['R_w']

    m_w = B_w.shape[1]

    # The optimization problem (common variables)
    problem = pc.Problem()
    solved = False
    problem.options.solver = 'mosek'
    problem.set_option('mosek_params', {'MSK_IPAR_NUM_THREADS': 1})

    v_Y = pc.SymmetricVariable('Y', (n, n)) # = X0*Q, acc. to De Persis
    v_U0M = pc.RealVariable('U0M', (m, n)) # product of U0 and M, which should be common for all systems
    v_Ms = [pc.RealVariable('M[{0}]'.format(trajId), (T, n)) for trajId in range(N)]
    # v_M = pc.RealVariable('M', (T, n))

    problem.set_objective(None)

    problem.add_constraint(v_Y >> 0)
    stab_lmis = list()
    # Trajectory-dependent constraints for each trajectory
    for trajId in range(N):

        U0 = trajectories[trajId]['U0']
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']

        v_M = v_Ms[trajId]

        stab00 = -v_Y
        stab10 = -S_w*v_M
        stab20 = X1*v_M
        stab30 = v_M
        stab11 = Q_w
        stab21 = B_w
        stab31 = np.zeros((T, m_w))
        stab22 = -v_Y
        stab32 = np.zeros((T, n))
        stab33 = -np.linalg.inv(R_w)
        stab = (
            (stab00 & stab10.T & stab20.T & stab30.T) //
            (stab10 & stab11   & stab21.T & stab31.T) //
            (stab20 & stab21   & stab22   & stab32.T) //
            (stab30 & stab31   & stab32   & stab33)
        )
        stab_lmis.append(stab)

        problem.add_constraint(stab << 0)
        problem.add_constraint(X0 * v_M == v_Y)
        problem.add_constraint(U0 * v_M == v_U0M)

    solution = None
    try:
        solver_verbosity = max(verbosity-1, 0)
        solution = problem.solve(verbosity=solver_verbosity)
    except pc.modeling.problem.SolutionFailure:
        if verbosity>0:
            print("Solution Failure occured")
        # pass # solution failure is raised for infeasible problems
    except ValueError:
        if verbosity>0:
            print("Value Error occured, problem will count as not solved")
    except Exception as e: #ill-conditioned matrices, unexpected exceptions
        # if verbosity>0:
        print(e)
        print("That's a caught exception, continuing with the rest of the programm")
        pass

    #print(problem.status)
    if solution is not None:
        if solution.status=='primal feasible' and solution.primalStatus=='optimal':
            solved = True
        else:
            if verbosity>0:
                print(f"Problem unsolved. solution.status: {solution.status}. solution.primalStatus: {solution.primalStatus}")
    else:
        if verbosity>0:
            print("Problem could not be solved")
        pass

    K = np.nan
    if solved:
        U0M = np.atleast_2d(v_U0M.value)
        Y = np.atleast_2d(v_Y.value)
        K = U0M @ np.linalg.inv(Y)
        #print('\nOptimal controller is found: \n{}\n'.format(K))
    return K


def robust_stabilization_scenario_slemma(trajectories, noiseInfo, perfInfo, verbosity=1):
    """
    robust LQR controller stabilizing each one of given trajectories and optimizing performance for them.
    Noise is assumed to be identically transferred to the states (B_w = I_nxn).

    @input: trajectories: list of dictionaries containing state sequence under 'state' and input sequence under 'input'. 'state' must have one more entry than 'input'
    @input: noiseInfo: dictionary containing matrices Phi_11, Phi_12, Phi_22 describing noise parameters and influence on the system
    @input: perfInfo: dictionary containing matrices Q, S, R describing performance metrics

    @ouput: K: designed controller. Nan if optimizer didn't succeed.

    Stabilization robust to disturbances is achieved via formulation acc. to:
    Waarde, Henk J. van, M. Kanat Camlibel, und Mehran Mesbahi. 2020.
    „From noisy data to feedback controllers: non-conservative design via a matrix S-lemma“.
    https://doi.org/10.48550/ARXIV.2006.00870.

    robustness to system uncertainty is achieved via scenario approach:
    Calafiore, G.C., und M.C. Campi. 2006.
    „The Scenario Approach to Robust Control Design“.
    IEEE Transactions on Automatic Control 51 (5): 742-53. https://doi.org/10.1109/TAC.2006.875041.

    """

    # Input parsing
    N = len(trajectories) # number of trajectories
    n = trajectories[0]['X0'].shape[0] # number of states
    m = trajectories[0]['U0'].shape[0] # number of inputs

    B_w = noiseInfo['B_w']
    m_w = noiseInfo['m_w']
    assert(B_w.all() == np.eye(B_w.shape[0]).all()) #van Waarde's paper works with this assumption

    Q = perfInfo['Q']
    S = perfInfo['S']
    R = perfInfo['R']

    # The optimization problem (common variables)
    problem = pc.Problem()
    solved = False
    problem.options.solver = 'mosek'
    problem.set_option('mosek_params', {'MSK_IPAR_NUM_THREADS': 1})

    # v_X = pc.SymmetricVariable('X', (m,m))
    v_P = pc.SymmetricVariable('P', (n,n)) # ~= Y?
    v_L = pc.RealVariable('L', (m,n)) # ~= U0M?
    v_a = pc.RealVariable('a')
    v_b = pc.RealVariable('b')

    # problem.set_objective(None)
    # problem.set_objective('min', pc.trace(v_P))
    # problem.options['dualize'] = True

    # problem.add_constraint(v_P >> 1e-4)
    problem.add_constraint(v_P >> 1e-6)
    problem.add_constraint(v_a >= 0)
    problem.add_constraint(v_b > 0)

    zeros_nn = pc.Constant('0nn', np.zeros((n,n)))
    zeros_mn = pc.Constant('0mn', np.zeros((m,n)))
    zeros_mm = pc.Constant('0mm', np.zeros((m,m)))

    stab_lin_00 = v_P - v_b * np.eye(n)
    stab_lin_10 = zeros_nn
    stab_lin_20 = zeros_mn
    stab_lin_30 = zeros_nn
    stab_lin_11 = -v_P
    stab_lin_21 = -v_L
    stab_lin_31 = zeros_nn
    stab_lin_22 = zeros_mm
    stab_lin_32 = v_L.T
    stab_lin_33 = v_P
    stab_lin = (
        (stab_lin_00   & stab_lin_10.T & stab_lin_20.T & stab_lin_30.T) //
        (stab_lin_10   & stab_lin_11   & stab_lin_21.T & stab_lin_31.T) //
        (stab_lin_20   & stab_lin_21   & stab_lin_22   & stab_lin_32.T) //
        (stab_lin_30   & stab_lin_31   & stab_lin_32   & stab_lin_33  )
    )
    # Trajectory-dependent constraints for each trajectory
    if noiseInfo['check_willems']:
            all_trajectories_persistently_exciting = True
    for trajId in range(N):

        U0 = trajectories[trajId]['U0'] # U_minus acc. to van Waarde
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']
        assumedBound = trajectories[trajId]['assumedBound']
        T = U0.shape[1] # length of one trajectory
        Phi_11 = assumedBound**2 * np.eye(m_w) * T
        Phi_12 = np.zeros((m_w, T))
        Phi_22 = -np.eye(T)
        Phi = np.block([[Phi_11  , Phi_12],
                        [Phi_12.T, Phi_22]])

        if noiseInfo['check_slater']:
            assert control_utils.check_gen_slater_condition(U0, X0, X1, Phi), "N has less than n positive eigenvalues"

        if noiseInfo['check_willems']:
            if not control_utils.check_persistent_excitation_willems(X0, U0):
                all_trajectories_persistently_exciting = False

        trajM = np.block([  [np.eye(n)      ,  X1],
                            [np.zeros((n,n)), -X0],
                            [np.zeros((m,n)), -U0],
                            [np.zeros((n,n)), np.zeros((n,T))]])
        stab_quad = trajM @ Phi @ trajM.T

        problem.add_constraint(stab_lin - v_a * stab_quad >> 0)

    if noiseInfo['check_willems'] and all_trajectories_persistently_exciting:
        problem.add_constraint(v_b == 0)
        if verbosity>0:
            print("All trajectories given fulfill the persistency of excitation. Setting beta to 0")


    solution = None
    try:
        solver_verbosity = max(verbosity-1, 0)
        solution = problem.solve(verbosity=solver_verbosity)
    except pc.modeling.problem.SolutionFailure:
        if verbosity>0:
            print("Solution Failure occured")
        # pass # solution failure is raised for infeasible problems
    except ValueError:
        if verbosity>0:
            print("Value Error occured, problem will count as not solved")
    except Exception as e: #ill-conditioned matrices, unexpected exceptions
        # if verbosity>0:
        print(e)
        print("That's a caught exception, continuing with the rest of the programm")
        pass

    #print(problem.status)
    if solution is not None:
        if solution.status=='primal feasible' and solution.primalStatus=='optimal':
            solved = True
        else:
            if verbosity>0:
                print(f"Problem unsolved. solution.status: {solution.status}. solution.primalStatus: {solution.primalStatus}")
    else:
        if verbosity>0:
            print("Problem could not be solved")
        pass

    K = np.nan
    if solved:
        L = np.atleast_2d(v_L.value)
        P = np.atleast_2d(v_P.value)
        K = L @ np.linalg.inv(P)
        if verbosity>0:
            a = v_a.value
            b = v_b.value
            print(' L: {0}\n P: {1}\n a: {2}\n b: {3}\n'.format(np.array_str(L, precision=3), np.array_str(P, precision=3), a, b))
            print('\nOptimal controller is found: \n{}\n'.format(K))
            print(f"DEBUG: eigenvalues of P: {np.linalg.eigvals(v_P)}")

    return K


def robust_lqr_scenario_slemma(trajectories, noiseInfo, perfInfo, verbosity=1):
    """
    robust LQR controller stabilizing each one of given trajectories and optimizing performance for them.
    Noise is assumed to be identically transferred to the states (B_w = I_nxn).

    @input: trajectories: list of dictionaries containing state sequence under 'state' and input sequence under 'input'. 'state' must have one more entry than 'input'
    @input: noiseInfo: dictionary containing matrices Phi_11, Phi_12, Phi_22 describing noise parameters and influence on the system
    @input: perfInfo: dictionary containing matrices Q, S, R describing performance metrics

    @ouput: K: designed controller. Nan if optimizer didn't succeed.

    Stabilization robust to disturbances is achieved via formulation acc. to:
    Waarde, Henk J. van, M. Kanat Camlibel, und Mehran Mesbahi. 2020.
    „From noisy data to feedback controllers: non-conservative design via a matrix S-lemma“.
    https://doi.org/10.48550/ARXIV.2006.00870.

    robustness to system uncertainty is achieved via scenario approach:
    Calafiore, G.C., und M.C. Campi. 2006.
    „The Scenario Approach to Robust Control Design“.
    IEEE Transactions on Automatic Control 51 (5): 742-53. https://doi.org/10.1109/TAC.2006.875041.

    optimal peformance is achieved by adapting:
    De Persis, Claudio, und Pietro Tesi. 2020.
    „Formulas for Data-Driven Control: Stabilization, Optimality, and Robustness“.
    IEEE Transactions on Automatic Control 65 (3): 909-24. https://doi.org/10.1109/TAC.2019.2959924.
    """

    # Input parsing
    N = len(trajectories) # number of trajectories
    n = trajectories[0]['X0'].shape[0] # number of states
    m = trajectories[0]['U0'].shape[0] # number of inputs

    B_w = noiseInfo['B_w']
    m_w = noiseInfo['m_w']
    assert(B_w.all() == np.eye(B_w.shape[0]).all()) #van Waarde's paper works with this assumption

    Q = perfInfo['Q']
    S = perfInfo['S']
    R = perfInfo['R']

    # The optimization problem (common variables)
    problem = pc.Problem()
    solved = False
    problem.options.solver = 'mosek'
    problem.set_option('mosek_params', {'MSK_IPAR_NUM_THREADS': 1})

    v_X = pc.SymmetricVariable('X', (m,m))
    v_P = pc.SymmetricVariable('P', (n,n)) # ~= Y?
    v_L = pc.RealVariable('L', (m,n)) # ~= U0M?
    v_a = pc.RealVariable('a')
    v_b = pc.RealVariable('b')

    problem.set_objective('min', pc.trace(Q*v_P) + pc.trace(v_X))
    # problem.set_objective(None)
    # problem.set_objective('max', v_b)
    problem.options['dualize'] = True

    perf00 = v_X
    perf01 = np.sqrt(R)  * v_L
    perf11 = v_P
    perf = (
        (perf00     & perf01) //
        (perf01.T   & perf11)
    )

    problem.add_constraint(perf >> 1e-4)
    problem.add_constraint(v_X >> 1e-4)
    problem.add_constraint(v_P >> 1e-4)
    problem.add_constraint(v_a >= 0)
    problem.add_constraint(v_b > 0)

    zeros_nn = pc.Constant('0nn', np.zeros((n,n)))
    zeros_mn = pc.Constant('0mn', np.zeros((m,n)))
    zeros_mm = pc.Constant('0mm', np.zeros((m,m)))

    stab_lin_00 = v_P - v_b * np.eye(n)
    stab_lin_10 = zeros_nn
    stab_lin_20 = zeros_mn
    stab_lin_30 = zeros_nn
    stab_lin_11 = -v_P
    stab_lin_21 = -v_L
    stab_lin_31 = zeros_nn
    stab_lin_22 = zeros_mm
    stab_lin_32 = v_L.T
    stab_lin_33 = v_P
    stab_lin = (
        (stab_lin_00   & stab_lin_10.T & stab_lin_20.T & stab_lin_30.T) //
        (stab_lin_10   & stab_lin_11   & stab_lin_21.T & stab_lin_31.T) //
        (stab_lin_20   & stab_lin_21   & stab_lin_22   & stab_lin_32.T) //
        (stab_lin_30   & stab_lin_31   & stab_lin_32   & stab_lin_33  )
    )
    # Trajectory-dependent constraints for each trajectory
    if noiseInfo['check_willems']:
            all_trajectories_persistently_exciting = True
    for trajId in range(N):

        U0 = trajectories[trajId]['U0'] # U_minus acc. to van Waarde
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']
        assumedBound = trajectories[trajId]['assumedBound']
        T = U0.shape[1] # length of one trajectory
        Phi_11 = assumedBound**2 * np.eye(m_w) * T
        Phi_12 = np.zeros((m_w, T))
        Phi_22 = -np.eye(T)
        Phi = np.block([[Phi_11  , Phi_12],
                        [Phi_12.T, Phi_22]])

        if noiseInfo['check_slater']:
            assert control_utils.check_gen_slater_condition(U0, X0, X1, Phi), "N has less than n positive eigenvalues"

        if noiseInfo['check_willems']:
            if not control_utils.check_persistent_excitation_willems(X0, U0):
                all_trajectories_persistently_exciting = False

        trajM = np.block([  [np.eye(n)      ,  X1],
                            [np.zeros((n,n)), -X0],
                            [np.zeros((m,n)), -U0],
                            [np.zeros((n,n)), np.zeros((n,T))]])
        stab_quad = trajM @ Phi @ trajM.T

        problem.add_constraint(stab_lin - v_a * stab_quad >> 0)

    if noiseInfo['check_willems'] and all_trajectories_persistently_exciting:
        problem.add_constraint(v_b == 0)
        if verbosity>0:
            print("All trajectories given fulfill the persistency of excitation. Setting beta to 0")


    solution = None
    try:
        solver_verbosity = max(verbosity-1, 0)
        solution = problem.solve(verbosity=solver_verbosity)
    except pc.modeling.problem.SolutionFailure:
        if verbosity>0:
            print("Solution Failure occured")
        # pass # solution failure is raised for infeasible problems
    except ValueError:
        if verbosity>0:
            print("Value Error occured, problem will count as not solved")
    except Exception as e: #ill-conditioned matrices, unexpected exceptions
        # if verbosity>0:
        print(e)
        print("That's a caught exception, continuing with the rest of the programm")
        pass

    #print(problem.status)
    if solution is not None:
        if solution.status=='primal feasible' and solution.primalStatus=='optimal':
            solved = True
        else:
            if verbosity>0:
                print(f"Problem unsolved. solution.status: {solution.status}. solution.primalStatus: {solution.primalStatus}")
    else:
        if verbosity>0:
            print("Problem could not be solved")
        pass

    K = np.nan
    if solved:
        L = np.atleast_2d(v_L.value)
        P = np.atleast_2d(v_P.value)
        K = L @ np.linalg.inv(P)
        if verbosity>0:
            a = v_a.value
            b = v_b.value
            print(' L: {0}\n P: {1}\n a: {2}\n b: {3}\n'.format(np.array_str(L, precision=3), np.array_str(P, precision=3), a, b))
            print('\nOptimal controller is found: \n{}\n'.format(K))

    return K



def robust_h2_scenario_slemma(trajectories, noiseInfo, perfInfo, verbosity=1):
    """
    robust controller stabilizing each one of given trajectories and optimizing H2 performance for them.
    Noise is assumed to be identically transferred to the states (B_w = I_nxn).

    @input: trajectories: list of dictionaries containing state sequence under 'state' and input sequence under 'input'. 'state' must have one more entry than 'input'
    @input: noiseInfo: dictionary containing matrices Phi_11, Phi_12, Phi_22 describing noise parameters and influence on the system
    @input: perfInfo: dictionary containing matrices Q, S, R describing performance metrics

    @ouput: K: designed controller. Nan if optimizer didn't succeed.

    robustness to disturbances and optimal performance is achieved via formulation acc. to:
    Waarde, Henk J. van, M. Kanat Camlibel, und Mehran Mesbahi. 2020.
    „From noisy data to feedback controllers: non-conservative design via a matrix S-lemma“.
    https://doi.org/10.48550/ARXIV.2006.00870.

    robustness to system uncertainty is achieved via scenario approach:
    Calafiore, G.C., und M.C. Campi. 2006.
    „The Scenario Approach to Robust Control Design“.
    IEEE Transactions on Automatic Control 51 (5): 742-53. https://doi.org/10.1109/TAC.2006.875041.
    """

    # Input parsing
    N = len(trajectories) # number of trajectories
    if N < noiseInfo['N_synth']:
        print("Received less trajectories than needed")
    n = trajectories[0]['X0'].shape[0] # number of states
    m = trajectories[0]['U0'].shape[0] # number of inputs

    B_w = noiseInfo['B_w']
    m_w = noiseInfo['m_w']
    assert(B_w.all() == np.eye(B_w.shape[0]).all()) #van Waarde's paper works with this assumption

    Q = perfInfo['Q']
    S = perfInfo['S']
    R = perfInfo['R']

    C = perfInfo['C']
    D = perfInfo['D']
    p = C.shape[0]
    assert D.shape[0] == p

    # The optimization problem (common variables)
    problem = pc.Problem()
    solved = False
    problem.options.solver = 'mosek'
    problem.set_option('mosek_params', {'MSK_IPAR_NUM_THREADS': 1})

    v_Y = pc.SymmetricVariable('Y', (n,n))
    v_Z = pc.SymmetricVariable('Z', (n,n))
    v_L = pc.RealVariable('L', (m,n))
    v_a = pc.RealVariable('a')
    v_b = pc.RealVariable('b')
    v_gamma_sqr = pc.RealVariable('γ²')
    problem.add_constraint(v_gamma_sqr > 1)

    problem.set_objective('min', v_gamma_sqr)
    # problem.options['dualize'] = True

    problem.add_constraint(v_Y >> 0)
    problem.add_constraint(v_a >= 0)
    problem.add_constraint(v_b > 0)

    zeros_nn = pc.Constant('0nn', np.zeros((n,n)))
    zeros_mn = pc.Constant('0mn', np.zeros((m,n)))
    zeros_mm = pc.Constant('0mm', np.zeros((m,m)))
    zeros_pn = pc.Constant('0pn', np.zeros((p,n)))
    zeros_pm = pc.Constant('0pm', np.zeros((p,m)))
    I_pp = pc.Constant('Ipp', np.eye(p))
    I_nn = pc.Constant('Inn', np.eye(n))

    C_YL = C * v_Y + D * v_L
    stab_lin_00 = v_Y - v_b * I_nn
    stab_lin_10 = zeros_nn
    stab_lin_20 = zeros_mn
    stab_lin_30 = zeros_nn
    stab_lin_40 = zeros_pn
    stab_lin_11 = zeros_nn
    stab_lin_21 = zeros_mn
    stab_lin_31 = v_Y
    stab_lin_41 = zeros_pn
    stab_lin_22 = zeros_mm
    stab_lin_32 = v_L.T
    stab_lin_42 = zeros_pm
    stab_lin_33 = v_Y
    stab_lin_43 = C_YL
    stab_lin_44 = I_pp
    stab_lin = (
        (stab_lin_00   & stab_lin_10.T & stab_lin_20.T & stab_lin_30.T & stab_lin_40.T) //
        (stab_lin_10   & stab_lin_11   & stab_lin_21.T & stab_lin_31.T & stab_lin_41.T) //
        (stab_lin_20   & stab_lin_21   & stab_lin_22   & stab_lin_32.T & stab_lin_42.T) //
        (stab_lin_30   & stab_lin_31   & stab_lin_32   & stab_lin_33   & stab_lin_43.T) //
        (stab_lin_40   & stab_lin_41   & stab_lin_42   & stab_lin_43   & stab_lin_44  )
    )
    lmi1 = (
        (v_Y  & C_YL.T) //
        (C_YL & I_pp)
    )
    lmi2 = (
        (v_Z  & I_nn) //
        (I_nn & v_Y)
    )
    problem.add_constraint(lmi1 >> 0)
    problem.add_constraint(lmi2 >> 0)
    problem.add_constraint(pc.trace(v_Z) < v_gamma_sqr)
    # Trajectory-dependent constraints for each trajectory

    if noiseInfo['check_willems']:
            all_trajectories_persistently_exciting = True
    for trajId in range(N):

        U0 = trajectories[trajId]['U0']
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']
        assumedBound = trajectories[trajId]['assumedBound']
        T = U0.shape[1] # length of one trajectory
        Phi_11 = assumedBound**2 * np.eye(m_w) * T
        Phi_12 = np.zeros((m_w, T))
        Phi_22 = -np.eye(T)
        Phi = np.block([[Phi_11  , Phi_12],
                        [Phi_12.T, Phi_22]])

        if noiseInfo['check_slater']:
            assert control_utils.check_gen_slater_condition(U0, X0, X1, Phi), "N has less than n positive eigenvalues"

        if noiseInfo['check_willems']:
            if not control_utils.check_persistent_excitation_willems(X0, U0):
                all_trajectories_persistently_exciting = False

        trajM = np.block([  [np.eye(n)      ,  X1],
                            [np.zeros((n,n)), -X0],
                            [np.zeros((m,n)), -U0],
                            [np.zeros((n,n)), np.zeros((n,T))],
                            [np.zeros((p,n)), np.zeros((p,T))]])
        stab_quad = trajM @ Phi @ trajM.T

        problem.add_constraint(stab_lin - v_a * stab_quad >> 0)

    if noiseInfo['check_willems'] and all_trajectories_persistently_exciting:
        problem.add_constraint(v_b == 0)
        print("All trajectories given fulfill the persistency of excitation. Setting beta to 0")


    solution = None
    try:
        solver_verbosity = max(verbosity-1, 0)
        solution = problem.solve(verbosity=solver_verbosity)
    except pc.modeling.problem.SolutionFailure:
        if verbosity>0:
            print("Solution Failure occured")
        # pass # solution failure is raised for infeasible problems
    except ValueError:
        if verbosity>0:
            print("Value Error occured, problem will count as not solved")
    except Exception as e: #ill-conditioned matrices, unexpected exceptions
        # if verbosity>0:
        print(e)
        print("That's a caught exception, continuing with the rest of the programm")
        pass

    #print(problem.status)
    if solution is not None:
        if solution.status=='primal feasible' and solution.primalStatus=='optimal':
            solved = True
        else:
            if verbosity>0:
                print(f"Problem unsolved. solution.status: {solution.status}. solution.primalStatus: {solution.primalStatus}")
    else:
        if verbosity>0:
            print("Problem could not be solved")
        pass

    K = np.nan
    if solved:
        L = np.atleast_2d(v_L.value)
        Y = np.atleast_2d(v_Y.value)
        K = L @ np.linalg.inv(Y)
        if verbosity>0:
            a = v_a.value
            b = v_b.value
            gamma_sqr = v_gamma_sqr.value
            print(' L: {0}\n Y: {1}\n a: {2}\n b: {3}\n γ²: {4}\n'.format(L, Y, a, b, gamma_sqr))
            print('\nOptimal controller is found: \n{}\n'.format(K))

    return K



def robust_hinf_scenario_slemma(trajectories, noiseInfo, perfInfo, verbosity=1):
    """
    robust controller stabilizing each one of given trajectories and optimizing H-Inf performance for them.
    Noise is assumed to be identically transferred to the states (B_w = I_nxn).

    @input: trajectories: list of dictionaries containing state sequence under 'state' and input sequence under 'input'. 'state' must have one more entry than 'input'
    @input: noiseInfo: dictionary containing matrices Phi_11, Phi_12, Phi_22 describing noise parameters and influence on the system
    @input: perfInfo: dictionary containing matrices Q, S, R describing performance metrics

    @ouput: K: designed controller. Nan if optimizer didn't succeed.

    robustness to disturbances and optimal performance is achieved via formulation acc. to:
    Waarde, Henk J. van, M. Kanat Camlibel, und Mehran Mesbahi. 2020.
    „From noisy data to feedback controllers: non-conservative design via a matrix S-lemma“.
    https://doi.org/10.48550/ARXIV.2006.00870.

    robustness to system uncertainty is achieved via scenario approach:
    Calafiore, G.C., und M.C. Campi. 2006.
    „The Scenario Approach to Robust Control Design“.
    IEEE Transactions on Automatic Control 51 (5): 742-53. https://doi.org/10.1109/TAC.2006.875041.
    """

    # Input parsing
    N = len(trajectories) # number of trajectories
    if N < noiseInfo['N_synth']:
        print("Received less trajectories than needed")
    n = trajectories[0]['X0'].shape[0] # number of states
    m = trajectories[0]['U0'].shape[0] # number of inputs

    B_w = noiseInfo['B_w']
    m_w = noiseInfo['m_w']
    assert(B_w.all() == np.eye(B_w.shape[0]).all()) #van Waarde's paper works with this assumption

    # Q = perfInfo['Q']
    # S = perfInfo['S']
    # R = perfInfo['R']

    C = perfInfo['C']
    D = perfInfo['D']
    p = C.shape[0]
    assert D.shape[0] == p

    # The optimization problem (common variables)
    problem = pc.Problem()
    solved = False
    problem.options.solver = 'mosek'
    problem.set_option('mosek_params', {'MSK_IPAR_NUM_THREADS': 1})

    v_Y = pc.SymmetricVariable('Y', (n,n))
    v_L = pc.RealVariable('L', (m,n))
    v_a = pc.RealVariable('a')
    # v_a = [pc.RealVariable('a{0}'.format(i)) for i in range(N)]
    v_b = pc.RealVariable('b')
    v_gamma_invsqr = pc.RealVariable('1/γ²')

    problem.set_objective('max', v_gamma_invsqr)
    # problem.options['dualize'] = True

    problem.add_constraint(v_Y >> 0)
    problem.add_constraint(v_a >= 0)
    problem.add_constraint(v_b > 0)
    problem.add_constraint(v_gamma_invsqr>1e-6)
    # problem.add_constraint(v_gamma_invsqr<1)
    # problem.add_constraint(v_gamma_invsqr==1/10000)

    zeros_nn = pc.Constant('0nn', np.zeros((n,n)))
    zeros_mn = pc.Constant('0mn', np.zeros((m,n)))
    zeros_mm = pc.Constant('0mm', np.zeros((m,m)))
    zeros_pn = pc.Constant('0pn', np.zeros((p,n)))
    zeros_pm = pc.Constant('0pm', np.zeros((p,m)))
    I_pp = pc.Constant('Ipp', np.eye(p))
    I_nn = pc.Constant('Inn', np.eye(n))

    C_YL = C * v_Y + D * v_L
    stab_lin_00 = v_Y - v_b * I_nn
    stab_lin_10 = zeros_nn
    stab_lin_20 = zeros_mn
    stab_lin_30 = zeros_nn
    stab_lin_40 = C_YL
    stab_lin_11 = zeros_nn
    stab_lin_21 = zeros_mn
    stab_lin_31 = v_Y
    stab_lin_41 = zeros_pn
    stab_lin_22 = zeros_mm
    stab_lin_32 = v_L.T
    stab_lin_42 = zeros_pm
    stab_lin_33 = v_Y - v_gamma_invsqr * I_nn
    stab_lin_43 = zeros_pn
    stab_lin_44 = I_pp
    stab_lin = (
        (stab_lin_00   & stab_lin_10.T & stab_lin_20.T & stab_lin_30.T & stab_lin_40.T) //
        (stab_lin_10   & stab_lin_11   & stab_lin_21.T & stab_lin_31.T & stab_lin_41.T) //
        (stab_lin_20   & stab_lin_21   & stab_lin_22   & stab_lin_32.T & stab_lin_42.T) //
        (stab_lin_30   & stab_lin_31   & stab_lin_32   & stab_lin_33   & stab_lin_43.T) //
        (stab_lin_40   & stab_lin_41   & stab_lin_42   & stab_lin_43   & stab_lin_44  )
    )
    problem.add_constraint(v_Y - v_gamma_invsqr * I_nn >> 0)
    # Trajectory-dependent constraints for each trajectory

    if noiseInfo['check_willems']:
            all_trajectories_persistently_exciting = True
    for trajId in range(N):

        U0 = trajectories[trajId]['U0']
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']
        assumedBound = trajectories[trajId]['assumedBound']
        T = U0.shape[1] # length of one trajectory
        Phi_11 = assumedBound**2 * np.eye(m_w) * T
        Phi_12 = np.zeros((m_w, T))
        Phi_22 = -np.eye(T)
        Phi = np.block([[Phi_11  , Phi_12],
                        [Phi_12.T, Phi_22]])

        if noiseInfo['check_slater']:
            assert control_utils.check_gen_slater_condition(U0, X0, X1, Phi), "N has less than n positive eigenvalues"

        if noiseInfo['check_willems']:
            if not control_utils.check_persistent_excitation_willems(X0, U0):
                all_trajectories_persistently_exciting = False

        trajM = np.block([  [np.eye(n)      ,  X1],
                            [np.zeros((n,n)), -X0],
                            [np.zeros((m,n)), -U0],
                            [np.zeros((n,n)), np.zeros((n,T))],
                            [np.zeros((p,n)), np.zeros((p,T))]])
        stab_quad = trajM @ Phi @ trajM.T

        problem.add_constraint(stab_lin - v_a * stab_quad >> 0)
        # problem.add_constraint(v_a[trajId] >= 0)

    if noiseInfo['check_willems'] and all_trajectories_persistently_exciting:
        problem.add_constraint(v_b == 0)
        if verbosity>0:
            print("All trajectories given fulfill the persistency of excitation. Setting beta to 0")


    solution = None
    try:
        solver_verbosity = max(verbosity-1, 0)
        solution = problem.solve(verbosity=solver_verbosity)
    except pc.modeling.problem.SolutionFailure:
        if verbosity>0:
            print("Solution Failure occured")
        # pass # solution failure is raised for infeasible problems
    except ValueError:
        if verbosity>0:
            print("Value Error occured, problem will count as not solved")
    except Exception as e: #ill-conditioned matrices, unexpected exceptions
        # if verbosity>0:
        print(e)
        print("That's a caught exception, continuing with the rest of the programm")
        pass

    #print(problem.status)
    if solution is not None:
        if solution.status=='primal feasible' and solution.primalStatus=='optimal':
            solved = True
        else:
            if verbosity>0:
                print(f"Problem unsolved. solution.status: {solution.status}. solution.primalStatus: {solution.primalStatus}")
    else:
        if verbosity>0:
            print("Problem could not be solved")
        pass

    K = np.nan
    if solved:
        L = np.atleast_2d(v_L.value)
        Y = np.atleast_2d(v_Y.value)
        K = L @ np.linalg.inv(Y)
        if verbosity>0:
            a = v_a.value
            b = v_b.value
            gamma_invsqr = v_gamma_invsqr.value
            gamma_sqr = 1/v_gamma_invsqr.value
            print(' L: {0}\n Y: {1}\n a: {2}\n b: {3}\n γ²: {4}\n 1/γ²: {5}\n'.format(np.array_str(L, precision=3), np.array_str(Y, precision=3), a, b, gamma_sqr, gamma_invsqr))
            # print('\nOptimal controller is found: \n{}\n'.format(K))
            # print(' L: {0}\n Y: {1}\n b: {2}\n γ²: {3}\n'.format(np.array_str(L, precision=3), np.array_str(Y, precision=3), b, gamma_sqr))
            # print('a: {0}'.format([v.value for v in v_a]))

    return K