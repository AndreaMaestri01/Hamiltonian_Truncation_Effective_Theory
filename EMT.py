import os
from tqdm import tqdm
import sys
from Htet import *

# Functions

def print_first_n(coeffs, states, n=10):
    pairs = list(zip(coeffs, states))
    # Sort by the absolute value of the coefficient in descending order
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[0]), reverse=True)
    # Print the first n
    for i, (coeff, state) in enumerate(pairs_sorted[:n]):
        print(f"Top {i+1}: coeff = {coeff:.2f}, state = {state}")


def load_config_by_index(i, db_path="Data/database.yaml"):

    config_name = f"config{i}"

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found in '{db_path}'")

    with open(db_path, 'r') as f:
        config_db = yaml.safe_load(f)

    if config_name not in config_db:
        raise KeyError(f"Configuration '{config_name}' is not present in the database.")

    config = config_db[config_name]
    print(f"Loaded configuration {config_name}: M={config['M']:.2f} Lambda/4π={config['Lambda'] / (4 * math.pi):.2f}, R·2π={config['R'] * 2 * math.pi:.2f}")
    return config


def load_eigenvectors(N_conf, N_eigens, Emax, moments, mode=None, base_dir="Data"):
    """
    Load eigenvectors from a given .npz file:
    - N_conf: configuration number (integer), e.g., 1 for 'config1'
    - Emax: energy truncation
    - moments: angular momentum or equivalent, used to create the folder name
    - n: eigenvector index
    - mode: parity ('even', 'odd' or None)
    - base_dir: base directory (default 'Data')
    """
    config_folder = f"config{N_conf}"
    moments_str = moments_to_filename(moments)

    folder = os.path.join(base_dir, config_folder, f"Moments_{moments_str}", "Eigenvectors")

    if mode == "even":
        folder += "_even"
    elif mode == "odd":
        folder += "_odd"

    filename = f"Eigenvec_Emax{Emax}_n{N_eigens}.npz"
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    data = np.load(path)
    #print(f"Loaded eigenvectors: Momentum = {moments} | Number of states = {len(data['coeffs'])}")
    return {
        "coeffs": data["coeffs"],
        "states": data["states"]
    }


def load_eigenvalues(N_conf, N_eigens, Emax, moments, mode=None, base_dir="Data"):

    config_folder = f"config{N_conf}"
    moments_str = moments_to_filename(moments)

    folder = os.path.join(base_dir, config_folder, f"Moments_{moments_str}", "Eigenvalues")

    if mode == "even":
        folder += "_even"
    elif mode == "odd":
        folder += "_odd"

    filename_V = f"Eigen_Emax{Emax}_V.txt"
    filename_VV = f"Eigen_Emax{Emax}_VV.txt"
    path_V = os.path.join(folder, filename_V)
    path_VV = os.path.join(folder, filename_VV)

    if not os.path.exists(path_V):
        raise FileNotFoundError(f"The file '{path_V}' does not exist.")
    if not os.path.exists(path_VV):
        raise FileNotFoundError(f"The file '{path_VV}' does not exist.")

    Eigen_V = np.loadtxt(path_V)[N_eigens]
    Eigen_VV = np.loadtxt(path_VV)[N_eigens]
    #print(f"Loaded eigenvalues:  Momentum = {moments} | Eigenvalue {N_eigens} ")

    return Eigen_V, Eigen_VV


def load_correction(N_conf, N_eigens, Emax, moments, mode=None, base_dir="Data"):
    config_folder = f"config{N_conf}"
    moments_str = moments_to_filename(moments)

    folder = os.path.join(base_dir, config_folder, f"Moments_{moments_str}", "Corrections")

    filename = f"Corr_Emax{Emax}.txt"
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    delta_m_2, delta_lambda_2 = np.loadtxt(path)

    #print(f"Loaded corrections")

    return delta_m_2, delta_lambda_2


def pad_state_to_python_int(state, delta: int) -> list[int]:
    """
    Converts each element of `state` (even if it is an np.int*) to a Python `int`,
    and then appends `delta` zeros (as ints) at the end.
    """
    # First: convert each element to built-in int
    converted = [int(x) for x in state]

    # Then append zeros if needed
    if delta <= 0:
        return converted
    return converted + [0] * delta


def N_state_analysis(states, X):
    parity_classes = set()

    for s, state in enumerate(states):
        parity = n_total(state) % 2
        parity_classes.add(parity)
    return parity_classes


def parity_description(parity_classes, X):
        if parity_classes == {0}:
            print(f"Momentum: {X} 	 Particles: even")
        elif parity_classes == {1}:
            print(f"Momentum: {X} 	 Particles: odd")
        else:
            print(f"Momentum: {X} 	 Particles: mixed")


def EMT_zero_analysis(N_particle_p, N_particle_q, P, Q):
    # Case in which both are homogeneous and equal: {0},{0} or {1},{1}
    if (N_particle_p == {0} and N_particle_q == {1}) or (N_particle_p == {1} and N_particle_q == {0}):
        return 0
    else:
        # Show the description for each momentum
        parity_description(N_particle_p, P)
        parity_description(N_particle_q, Q)


# Operators

def low_k1_k2(Emax, M, R, state, state_index, ix_list, k1, k2):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]

    fac = 0
    pos = None

    if temp_state[ix_k1] != 0:
        factor1 = math.sqrt(temp_state[ix_k1])
        temp_state[ix_k1] -= 1
        if temp_state[ix_k2] != 0:
            fac = factor1 * math.sqrt(temp_state[ix_k2])
            temp_state[ix_k2] -= 1
            pos = state_index.get(tuple(temp_state))
    return fac, pos


def rai_k1_k2(Emax, M, R, state, state_index, ix_list, k1, k2):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]

    fac = 0
    pos = None

    factor1 = math.sqrt(temp_state[ix_k1] + 1)
    temp_state[ix_k1] += 1
    if state_energy(temp_state, M, R) < Emax:
        fac = factor1 * math.sqrt(temp_state[ix_k2] + 1)
        temp_state[ix_k2] += 1
        if state_energy(temp_state, M, R) < Emax:
            pos = state_index.get(tuple(temp_state))
    return fac, pos


def rai_k1_low_k2(Emax, M, R, state, state_index, ix_list, k1, k2):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]

    fac = 0
    pos = None
    if temp_state[ix_k2] != 0:
        factor1 = math.sqrt(temp_state[ix_k2])
        temp_state[ix_k2] -= 1

        fac = factor1 * math.sqrt(temp_state[ix_k1] + 1)
        temp_state[ix_k1] += 1
        if state_energy(temp_state, M, R) < Emax:
            pos = state_index.get(tuple(temp_state))
    return fac, pos


def rai_k2_low_k1(Emax, M, R, state, state_index, ix_list, k1, k2):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]

    fac = 0
    pos = None
    if temp_state[ix_k1] != 0:
        factor1 = math.sqrt(temp_state[ix_k1])
        temp_state[ix_k1] -= 1

        fac = factor1 * math.sqrt(temp_state[ix_k2] + 1)
        temp_state[ix_k2] += 1
        if state_energy(temp_state, M, R) < Emax:
            pos = state_index.get(tuple(temp_state))
    return fac, pos


def low_k1_k2_k3_k4(Emax, M, R, state, state_index, ix_list, k1, k2, k3, k4):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]
    ix_k4 = ix_list[k4]
    fac = 0
    pos = None

    if temp_state[ix_k1] != 0:
        factor1 = math.sqrt(temp_state[ix_k1])
        temp_state[ix_k1] -= 1
        if temp_state[ix_k2] != 0:
            factor2 = factor1 * math.sqrt(temp_state[ix_k2])
            temp_state[ix_k2] -= 1
            if temp_state[ix_k3] != 0:
                factor3 = factor2 * math.sqrt(temp_state[ix_k3])
                temp_state[ix_k3] -= 1
                if temp_state[ix_k4] != 0:
                    fac = factor3 * math.sqrt(temp_state[ix_k4])
                    temp_state[ix_k4] -= 1
                    pos = state_index.get(tuple(temp_state))
    return fac, pos


def rai_k1_low_k2_k3_k4(Emax, M, R, state, state_index, ix_list, k1, k2, k3, k4):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]
    ix_k4 = ix_list[k4]
    fac = 0
    pos = None
    if temp_state[ix_k4] != 0:
        factor1 = math.sqrt(temp_state[ix_k4])
        temp_state[ix_k4] -= 1
        if temp_state[ix_k3] != 0:
            factor2 = factor1 * math.sqrt(temp_state[ix_k3])
            temp_state[ix_k3] -= 1
            if temp_state[ix_k2] != 0:
                factor3 = factor2 * math.sqrt(temp_state[ix_k2])
                temp_state[ix_k2] -= 1
                fac = factor3 * math.sqrt(temp_state[ix_k1] + 1)
                temp_state[ix_k1] += 1
                if state_energy(temp_state, M, R) < Emax:
                    pos = state_index.get(tuple(temp_state))
    return fac, pos


def rai_k1_k2_low_k3_k4(Emax, M, R, state, state_index, ix_list, k1, k2, k3, k4):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]
    ix_k4 = ix_list[k4]
    fac = 0
    pos = None
    if temp_state[ix_k4] != 0:
        factor1 = math.sqrt(temp_state[ix_k4])
        temp_state[ix_k4] -= 1
        if temp_state[ix_k3] != 0:
            factor2 = factor1 * math.sqrt(temp_state[ix_k3])
            temp_state[ix_k3] -= 1

            factor3 = factor2 * math.sqrt(temp_state[ix_k2] + 1)
            temp_state[ix_k2] += 1
            if state_energy(temp_state, M, R) < Emax:
                fac = factor3 * math.sqrt(temp_state[ix_k1] + 1)
                temp_state[ix_k1] += 1
                if state_energy(temp_state, M, R) < Emax:
                    pos = state_index.get(tuple(temp_state))
    return fac, pos


def rai_k1_k2_k3_low_k4(Emax, M, R, state, state_index, ix_list, k1, k2, k3, k4):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]
    ix_k4 = ix_list[k4]
    fac = 0
    pos = None
    if temp_state[ix_k4] != 0:
        factor1 = math.sqrt(temp_state[ix_k4])
        temp_state[ix_k4] -= 1

        factor2 = factor1 * math.sqrt(temp_state[ix_k3] + 1)
        temp_state[ix_k3] += 1
        if state_energy(temp_state, M, R) < Emax:
            factor3 = factor2 * math.sqrt(temp_state[ix_k2] + 1)
            temp_state[ix_k2] += 1
            if state_energy(temp_state, M, R) < Emax:
                fac = factor3 * math.sqrt(temp_state[ix_k1] + 1)
                temp_state[ix_k1] += 1
                if state_energy(temp_state, M, R) < Emax:
                    pos = state_index.get(tuple(temp_state))
    return fac, pos


def rai_k1_k2_k3_k4(Emax, M, R, state, state_index, ix_list, k1, k2, k3, k4):
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]
    ix_k4 = ix_list[k4]
    fac = 0
    pos = None

    factor1 = math.sqrt(temp_state[ix_k4] + 1)
    temp_state[ix_k4] += 1
    if state_energy(temp_state, M, R) < Emax:
        factor2 = factor1 * math.sqrt(temp_state[ix_k3] + 1)
        temp_state[ix_k3] += 1
        if state_energy(temp_state, M, R) < Emax:
            factor3 = factor2 * math.sqrt(temp_state[ix_k2] + 1)
            temp_state[ix_k2] += 1
            if state_energy(temp_state, M, R) < Emax:
                fac = factor3 * math.sqrt(temp_state[ix_k1] + 1)
                temp_state[ix_k1] += 1
                if state_energy(temp_state, M, R) < Emax:
                    pos = state_index.get(tuple(temp_state))
    return fac, pos


def precompute_terms_T01_DivStates(states1, coeffs1, coeffs2, omega_list1, state_index2, ix_list1, lmax1, Emax1, M, R, term_func):
    value = 0
    for i, state in enumerate(states1):
        c_i = coeffs1[i]

        for k1 in range(-lmax1, lmax1 + 1):
            for k2 in range(-lmax1, lmax1 + 1):

                    if term_func == low_k1_k2 or term_func == rai_k1_k2:
                        if omega_list1[k1] + omega_list1[k2] >= Emax1:
                            continue

                    if term_func == rai_k1_low_k2 or term_func == rai_k2_low_k1:
                        if omega_list1[k1] >= Emax1 and omega_list1[k2] >= Emax1:
                            continue

                    fac, pos = term_func(Emax1, M, R, state, state_index2, ix_list1, k1, k2)
                    if pos is not None and fac != 0:
                        c_j = coeffs2[pos]

                        factor = c_i * c_j * fac * (- 1 / (4 * math.pi * R)) * (math.sqrt(omega_list1[k1] / omega_list1[k2])) * (k2 / R)
                        value += factor
    return value


def compute_T01_terms(
    states_i_p, coeffs_i_p, coeffs_i_q,
    omega_list_p, state_index_q, ix_list_p,
    lmax_p, Emaxx, M, R,
    term_funcs_list=None
):
    """
    Compute the four components of T01 (low, rai, rai_low, low_rai)
    by calling `precompute_terms_T01_DivStates` four times with the corresponding `term_func`.

    Returns the tuple:
      (low_val, rai_val, rai_low_val, low_rai_val).
    """

    # 1) Define default term_funcs if none are specified externally
    if term_funcs_list is None:
        term_funcs_list = [low_k1_k2,
                           rai_k1_k2,
                           rai_k1_low_k2,
                           rai_k2_low_k1]

    # Check: the list must contain exactly 4 functions
    if len(term_funcs_list) != 4:
        raise ValueError("compute_T01_terms: term_funcs_list must contain exactly 4 elements.")

    # 2) Extract each term_func from the list
    low_func, rai_func, rai_low_func, low_rai_func = term_funcs_list

    # 3) Calls to precompute_terms_T01_DivStates
    low_val = precompute_terms_T01_DivStates(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, low_func
    )

    rai_val = precompute_terms_T01_DivStates(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, rai_func
    )

    rai_low_val = precompute_terms_T01_DivStates(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, rai_low_func
    )

    low_rai_val = precompute_terms_T01_DivStates(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, low_rai_func
    )

    # 4) Return the four results as a tuple
    return low_val, rai_val, rai_low_val, low_rai_val


def T01_0(N_conf, Emax, P, Q, Eigen_i):
    config = load_config_by_index(N_conf)
    Lambda = config["Lambda"]
    M = config["M"]
    R = config["R"]

    p = P / R
    q = Q / R

    Emax_p = np.sqrt((Emax) ** 2 + (p) ** 2)
    Emax_q = np.sqrt((Emax) ** 2 + (q) ** 2)
    Emaxx = max(Emax_p, Emax_q)

    N = 4.0
    if M**2 < (Emax_p**2) / N:
        lmax_p = int(math.floor(math.sqrt(R**2 * (Emax_p**2 / N - M**2))))
    else:
        lmax_p = 0
    print(f"LmaxP: {lmax_p}")
    if M**2 < (Emax_q**2) / N:
        lmax_q = int(math.floor(math.sqrt(R**2 * (Emax_q**2 / N - M**2))))
    else:
        lmax_q = 0
    print(f"LmaxQ: {lmax_q}")

    # Momentum 0
    Evac_0 = load_eigenvalues(N_conf, 0, Emax, moments=[0])[1]
    E1_0 = load_eigenvalues(N_conf, 1, Emax, moments=[0])[1]
    m_gap = E1_0 - Evac_0

    # Momentum P != 0
    E1_p = load_eigenvalues(N_conf, 0, Emax, moments=[P])[1]
    Evac_p = E1_p - np.sqrt(m_gap**2 + (p) ** 2)
    Ei_p = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[P])[1]
    results_i_p = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[P])
    coeffs_i_p = results_i_p["coeffs"]
    states_i_p = results_i_p["states"]

    # Momentum Q != 0
    E1_q = load_eigenvalues(N_conf, 0, Emax, moments=[Q])[1]
    Evac_q = E1_q - np.sqrt(m_gap**2 + (q) ** 2)
    Ei_q = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[Q])[1]
    results_i_q = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[Q])
    coeffs_i_q = results_i_q["coeffs"]
    states_i_q = results_i_q["states"]

    lmaxx = max(lmax_p, lmax_q)
    N_comune = 2 * lmaxx + 1
    delta_p = N_comune - (2 * lmax_p + 1)
    delta_q = N_comune - (2 * lmax_q + 1)

    # padded_states_i_p.append(padded)
    padded_states_i_p = np.array([pad_state_to_python_int(s, delta_p) for s in states_i_p])
    padded_states_i_q = np.array([pad_state_to_python_int(s, delta_q) for s in states_i_q])

    state_index_p = {tuple(ps): i for i, ps in enumerate(padded_states_i_p)}
    state_index_q = {tuple(ps): i for i, ps in enumerate(padded_states_i_q)}

    omega_list_max = {k: omega(k, M, R) for k in range(-lmaxx, lmaxx + 1)}
    ix_list_max = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmaxx, lmaxx + 1)}

    low, rai, rai_low, low_rai = compute_T01_terms(
        padded_states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_max, state_index_q, ix_list_max,
        lmaxx, Emaxx, R
    )

    T01 = low + rai - rai_low - low_rai

    return T01


def precompute_terms_T00_all_DivStates_separate(
    states1, coeffs1, coeffs2,
    omega_list1, state_index2, ix_list1, lmax1, Emax1,
    M, R, Lambda, delta_m_2, delta_lambda, term_func
):
    Mass_2 = M**2 + delta_m_2

    Mass_sum = 0.0
    Time_sum = 0.0
    Space_sum = 0.0

    for i, state in enumerate(states1):
        c_i = coeffs1[i]
        for k1 in range(-lmax1, lmax1 + 1):
            for k2 in range(-lmax1, lmax1 + 1):

                if term_func in (low_k1_k2, rai_k1_k2):
                    if omega_list1[k1] + omega_list1[k2] >= Emax1:
                        continue

                if term_func in (rai_k1_low_k2, rai_k2_low_k1):
                    if omega_list1[k1] >= Emax1 and omega_list1[k2] >= Emax1:
                        continue

                fac, pos = term_func(Emax1, M, R, state, state_index2, ix_list1, k1, k2)
                if pos is not None and fac != 0:
                    c_j = coeffs2[pos]

                    time_fac = (-1 / (8 * math.pi * R)) * math.sqrt(omega_list1[k1] * omega_list1[k2])
                    space_fac = (-1 / (8 * math.pi * R)) * (1.0 / math.sqrt(omega_list1[k1] * omega_list1[k2])) * ((k1 * k2) / (R**2))
                    mass_fac = (1 / (8 * math.pi * R)) * (1.0 / math.sqrt(omega_list1[k1] * omega_list1[k2])) * Mass_2

                    if term_func in (low_k1_k2, rai_k1_k2):
                        mass_contrib = c_i * c_j * fac * mass_fac
                        time_contrib = c_i * c_j * fac * time_fac
                        space_contrib = c_i * c_j * fac * space_fac
                    else:
                        mass_contrib = c_i * c_j * fac * mass_fac
                        time_contrib = c_i * c_j * fac * (- time_fac)
                        space_contrib = c_i * c_j * fac * (- space_fac)

                    Mass_sum += mass_contrib
                    Time_sum += time_contrib
                    Space_sum += space_contrib

    return Mass_sum, Time_sum, Space_sum


def precompute_terms_T00_lambda_DivStates_separate(
    states1, coeffs1, coeffs2,
    omega_list1, state_index2, ix_list1, lmax1, Emax1,
    M, R, Lambda, delta_m_2, delta_lambda, term_func
):
    Lambda_eff = Lambda + delta_lambda

    Lambda_sum = 0.0
    for i, state in enumerate(states1):
        c_i = coeffs1[i]
        for k1 in range(-lmax1, lmax1 + 1):
            for k2 in range(-lmax1, lmax1 + 1):
                for k3 in range(-lmax1, lmax1 + 1):
                    for k4 in range(-lmax1, lmax1 + 1):

                        if term_func in (low_k1_k2_k3_k4, rai_k1_k2_k3_k4):
                            if omega_list1[k1] + omega_list1[k2] + omega_list1[k3] + omega_list1[k4] >= Emax1:
                                continue
                        if term_func == rai_k1_low_k2_k3_k4:
                            if omega_list1[k1] >= Emax1 and (omega_list1[k2] + omega_list1[k3] + omega_list1[k4]) >= Emax1:
                                continue
                        if term_func == rai_k1_k2_k3_low_k4:
                            if (omega_list1[k1] + omega_list1[k2] + omega_list1[k3]) >= Emax1 and omega_list1[k4] >= Emax1:
                                continue
                        if term_func == rai_k1_k2_low_k3_k4:
                            if (omega_list1[k1] + omega_list1[k2]) >= Emax1 and (omega_list1[k3] + omega_list1[k4]) >= Emax1:
                                continue
                        fac, pos = term_func(Emax1, M, R, state, state_index2, ix_list1, k1, k2, k3, k4)
                        if pos is not None and fac != 0:
                            c_j = coeffs2[pos]
                            factor = (c_i * c_j * fac) * (Lambda_eff / 24) * (1.0 / (2 * math.pi * R)**2) * \
                                     (1.0 / math.sqrt(2 * omega_list1[k1])) * \
                                     (1.0 / math.sqrt(2 * omega_list1[k2])) * \
                                     (1.0 / math.sqrt(2 * omega_list1[k3])) * \
                                     (1.0 / math.sqrt(2 * omega_list1[k4]))
                            Lambda_sum += factor
    return Lambda_sum


def compute_two_point_sums(
    states_i_p, coeffs_i_p, coeffs_i_q,
    omega_list_p, state_index_q, ix_list_p,
    lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda
):
    """
    Compute the total Mass, Time, and Space contributions from the '2-point' diagrams 
    (low_k1_k2, rai_k1_k2, rai_k1_low_k2, rai_k2_low_k1), by calling four times
    `precompute_terms_T00_all_DivStates_separate` with the respective `term_func`.

    Parameters:
    - states_i_p, coeffs_i_p:  states and coefficients for P
    - coeffs_i_q:              coefficients for Q
    - omega_list_p:            dict {k: omega(k,M,R)} for k ∈ [-lmax_p, +lmax_p]
    - state_index_q:           mapping {tuple(state): index} for Q
    - ix_list_p:               mapping for the i-th state index in P
    - lmax_p:                  angular index cutoff for P
    - Emaxx:                   energy cutoff (max between p and q)
    - R, M, Lambda:            physical parameters (radius, mass, quartic coupling)
    - delta_m_2:               mass correction (M² → M²+δm²)
    - delta_lambda:            quartic correction
    """

    # 1) Call with term_func = low_k1_k2
    Mass_low, Time_low, Space_low = precompute_terms_T00_all_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        low_k1_k2
    )

    # 2) Call with term_func = rai_k1_k2
    Mass_rai, Time_rai, Space_rai = precompute_terms_T00_all_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        rai_k1_k2
    )

    # 3) Call with term_func = rai_k1_low_k2
    Mass_rai_low, Time_rai_low, Space_rai_low = precompute_terms_T00_all_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        rai_k1_low_k2
    )

    # 4) Call with term_func = rai_k2_low_k1
    Mass_low_rai, Time_low_rai, Space_low_rai = precompute_terms_T00_all_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        rai_k2_low_k1
    )

    # 5) Overall sum of the contributions
    Mass_sum_total = Mass_low + Mass_rai + Mass_rai_low + Mass_low_rai
    Time_sum_total = Time_low + Time_rai + Time_rai_low + Time_low_rai
    Space_sum_total = Space_low + Space_rai + Space_rai_low + Space_low_rai

    return Mass_sum_total, Time_sum_total, Space_sum_total


def compute_lambda_sum_total(
    states_i_p, coeffs_i_p, coeffs_i_q,
    omega_list_p, state_index_q, ix_list_p,
    lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda
):
    """
    Compute the total quartic contribution `Lambda_sum_total` by combining
    the five 4-point diagrams: 
      - low_k1_k2_k3_k4
      - rai_k1_k2_k3_k4
      - rai_k1_low_k2_k3_k4
      - rai_k1_k2_k3_low_k4
      - rai_k1_k2_low_k3_k4

    Each call to `precompute_terms_T00_lambda_DivStates_separate` returns
    a single sum (the value of the quartic contribution). We combine them
    with coefficients 1, 1, 4, 4, 6, in the appropriate order.
    """

    # 1) Contribution “low_k1_k2_k3_k4”
    contrib_low_4 = precompute_terms_T00_lambda_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        low_k1_k2_k3_k4
    )

    # 2) Contribution “rai_k1_k2_k3_k4”
    contrib_rai_4 = precompute_terms_T00_lambda_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        rai_k1_k2_k3_k4
    )

    # 3) Contribution “rai_k1_low_k2_k3_k4”
    contrib_rai_low_4 = precompute_terms_T00_lambda_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        rai_k1_low_k2_k3_k4
    )

    # 4) Contribution “rai_k1_k2_k3_low_k4”
    contrib_rai_low_end = precompute_terms_T00_lambda_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        rai_k1_k2_k3_low_k4
    )

    # 5) Contribution “rai_k1_k2_low_k3_k4”
    contrib_rai_low_mid = precompute_terms_T00_lambda_DivStates_separate(
        states_i_p, coeffs_i_p, coeffs_i_q,
        omega_list_p, state_index_q, ix_list_p,
        lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda,
        rai_k1_k2_low_k3_k4
    )

    # 6) Final combination: 1*low_4 + 1*rai_4 + 4*rai_low_4 + 4*rai_low_end + 6*rai_low_mid
    Lambda_sum_total = (
        contrib_low_4
      + contrib_rai_4
      + 4   * contrib_rai_low_4
      + 4   * contrib_rai_low_end
      + 6   * contrib_rai_low_mid
    )

    return Lambda_sum_total


def compute_T_PQ(N_conf, Emax, P, Q, Eigen_i, which=("T01", "T00", "T11"), save_mode="yes"):
    config = load_config_by_index(N_conf)
    delta_m_2, delta_lambda = load_correction(N_conf, Eigen_i - 1, Emax, [0])

    Lambda = config["Lambda"]
    M = config["M"]
    R = config["R"]

    p = P / R
    q = Q / R

    Emax_p = np.sqrt((Emax) ** 2 + (p) ** 2)
    Emax_q = np.sqrt((Emax) ** 2 + (q) ** 2)
    Emaxx = max(Emax_p, Emax_q)

    N = 4.0
    if M**2 < (Emax_p**2) / N:
        lmax_p = int(math.floor(math.sqrt(R**2 * (Emax_p**2 / N - M**2))))
    else:
        lmax_p = 0

    if M**2 < (Emax_q**2) / N:
        lmax_q = int(math.floor(math.sqrt(R**2 * (Emax_q**2 / N - M**2))))
    else:
        lmax_q = 0

    # Momentum 0
    Evac_0 = load_eigenvalues(N_conf, 0, Emax, moments=[0])[1]
    E1_0 = load_eigenvalues(N_conf, 1, Emax, moments=[0])[1]
    m_gap = E1_0 - Evac_0

    # Momentum P != 0
    E1_p = load_eigenvalues(N_conf, 0, Emax, moments=[P])[1]
    Evac_p = E1_p - np.sqrt(m_gap**2 + (p) ** 2)
    Ei_p = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[P])[1]
    results_i_p = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[P])
    coeffs_i_p = results_i_p["coeffs"]
    states_i_p = results_i_p["states"]
    N_particle_p = N_state_analysis(states_i_p, P)

    # Momentum Q != 0
    E1_q = load_eigenvalues(N_conf, 0, Emax, moments=[Q])[1]
    Evac_q = E1_q - np.sqrt(m_gap**2 + (q) ** 2)
    Ei_q = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[Q])[1]
    results_i_q = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[Q])
    coeffs_i_q = results_i_q["coeffs"]
    states_i_q = results_i_q["states"]
    N_particle_q = N_state_analysis(states_i_q, Q)

    if EMT_zero_analysis(N_particle_p, N_particle_q, P, Q) == 0:
        results = {}
        results["T01"] = 0
        results["T00"] = 0
        results["T11"] = 0
        raise ValueError(f"[ALERT] States with different particle numbers T=0")

    lmaxx = max(lmax_p, lmax_q)
    N_comune = 2 * lmaxx + 1
    delta_p = N_comune - (2 * lmax_p + 1)
    delta_q = N_comune - (2 * lmax_q + 1)

    # padded_states_i_p.append(padded)
    padded_states_i_p = np.array([pad_state_to_python_int(s, delta_p) for s in states_i_p])
    padded_states_i_q = np.array([pad_state_to_python_int(s, delta_q) for s in states_i_q])

    state_index_p = {tuple(ps): i for i, ps in enumerate(padded_states_i_p)}
    state_index_q = {tuple(ps): i for i, ps in enumerate(padded_states_i_q)}

    omega_list_max = {k: omega(k, M, R) for k in range(-lmaxx, lmaxx + 1)}
    ix_list_max = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmaxx, lmaxx + 1)}

    results = {}

    if "T01" in which:
        low, rai, rai_low, low_rai = compute_T01_terms(
            padded_states_i_p, coeffs_i_p, coeffs_i_q,
            omega_list_max, state_index_q, ix_list_max,
            lmax_p, Emaxx, M, R
        )
        T01_val = low + rai - rai_low - low_rai
        results["T01"] = T01_val
        if P == Q:
            print(f"Momentum accuracy: {((P/R - (T01_val*(2*np.pi*R)))/P)*100} %")
    else:
        results["T01"] = np.nan

    if ("T00" in which) or ("T11" in which):
        Mass_sum_total, Time_sum_total, Space_sum_total = compute_two_point_sums(
            padded_states_i_p, coeffs_i_p, coeffs_i_q,
            omega_list_max, state_index_q, ix_list_max,
            lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda
        )

        Lambda_sum_total = compute_lambda_sum_total(
            padded_states_i_p, coeffs_i_p, coeffs_i_q,
            omega_list_max, state_index_q, ix_list_max,
            lmax_p, Emaxx, M, R, Lambda, delta_m_2, delta_lambda
        )

        if "T00" in which:
            T00_val = Mass_sum_total + Time_sum_total + Space_sum_total + Lambda_sum_total
            results["T00"] = T00_val
            if P == Q:
                print(f"Energy accuracy: {((Ei_p - (T00_val*(2*np.pi*R)))/Ei_p)*100} %")
        else:
            results["T00"] = np.nan

        if "T11" in which:
            T11_val = Time_sum_total + Space_sum_total - Mass_sum_total - Lambda_sum_total
            results["T11"] = T11_val
        else:
            results["T11"] = np.nan
    else:
        results["T00"] = np.nan
        results["T11"] = np.nan

    if save_mode not in {"yes", "no"}:
        raise ValueError(f"Invalid save_mode: '{save_mode}'. Must be either 'yes' or 'no'.")
    if save_mode == "yes":
        BASE_DIR = "/home/andrea-maestri/University/Tesi_Magistrale/Code/HTET/EMT/Points"
        DATA_SRC = "/home/andrea-maestri/University/Tesi_Magistrale/Code/HTET/Data/database.yaml"
        # 1) Folders config{N_conf}/Emax_{Emax}
        config_dir = os.path.join(BASE_DIR, f"config{N_conf}")
        emax_dir = os.path.join(config_dir, f"Emax_{Emax}")
        os.makedirs(emax_dir, exist_ok=True)

        # 2) Copy or create database.yaml in BASE_DIR
        points_db = os.path.join(BASE_DIR, "database.yaml")
        if not os.path.isfile(points_db):
            # Read the entire database
            with open(DATA_SRC, 'r') as f:
                full_db = yaml.safe_load(f)

            key = f"config{N_conf}"
            if key not in full_db:
                raise KeyError(f"'{key}' not found in {DATA_SRC}")

            # Write only the requested section
            new_db = {key: full_db[key]}
            with open(points_db, 'w') as f:
                yaml.dump(new_db, f, default_flow_style=False)

        # 3) Save or update FormFactor.txt
        file_path = os.path.join(emax_dir, "FormFactor.txt")
        header = "P Q R T00 T11 T01"
        new_row = np.array([P, Q, R, results["T00"], results["T11"], results["T01"]], dtype=float)

        # Load existing entries into a dictionary
        data_dict = {}
        if os.path.isfile(file_path):
            try:
                existing = np.loadtxt(file_path, skiprows=1)
                if existing.ndim == 1:
                    existing = existing.reshape(1, -1)
                else:
                    existing = existing.reshape(-1, existing.shape[-1])
                for row in existing:
                    data_dict[(row[0], row[1])] = row
            except Exception as e:
                print(f"[WARNING] Error loading existing file: {e}")
                data_dict = {}

        # Add or overwrite
        data_dict[(P, Q)] = new_row

        # Rewrite sorted
        all_rows = np.vstack([data_dict[k] for k in sorted(data_dict.keys())])
        np.savetxt(file_path, all_rows, header=header, comments='', fmt="%.10e")
        print(f"[INFO] Data saved to {file_path}")
    return results


def compute_T_M0(N_conf, Emax, P, Q, Eigen_i, which=("T01", "T00", "T11")):

    config = load_config_by_index(N_conf)
    delta_m_2, delta_lambda = load_correction(N_conf, Eigen_i, Emax, [0])  # or -1?
    EigV, EigVV = load_eigenvalues(N_conf, Eigen_i, Emax, moments=[0])
    Lambda = config["Lambda"]
    lam_4pi = Lambda / (4 * np.pi)
    M = config["M"]
    R = config["R"]
    result = load_eigenvectors(N_conf, Eigen_i, Emax, moments=[0])
    coeffs = result["coeffs"]
    states = result["states"]

    N = 4.0
    if M**2 < (Emax**2) / N:
        lmax = int(math.floor(math.sqrt(R**2 * (Emax**2 / N - M**2))))
    else:
        lmax = 0
    # Creation of omega_list
    omega_list = {k: omega(k, M, R) for k in range(-lmax, lmax + 1)}
    ix_list = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmax, lmax + 1)}
    state_index = {tuple(s): i for i, s in enumerate(states)}

    results = {}

    if "T01" in which:

        low, rai, rai_low, low_rai = compute_T01_terms(
            states, coeffs, coeffs,
            omega_list, state_index, ix_list,
            lmax, Emax, M, R,
        )
        T01_val = low + rai - rai_low - low_rai
        results["T01"] = T01_val
        print(f"Momentum accuracy: {((P/R - (T01_val*(2*np.pi*R)))/P)*100} %")

    if ("T00" in which) or ("T11" in which):
        Mass_sum_total, Time_sum_total, Space_sum_total = compute_two_point_sums(
            states, coeffs, coeffs,
            omega_list, state_index, ix_list,
            lmax, Emax, M, R, Lambda, delta_m_2, delta_lambda
        )

        Lambda_sum_total = compute_lambda_sum_total(
            states, coeffs, coeffs,
            omega_list, state_index, ix_list,
            lmax, Emax, M, R, Lambda, delta_m_2, delta_lambda
        )

        if "T00" in which:
            T00_val = Mass_sum_total + Time_sum_total + Space_sum_total + Lambda_sum_total
            results["T00"] = T00_val
            print(f"Energy accuracy: {((EigVV - (T00_val*(2*np.pi*R)))/EigVV)*100} %")

        if "T11" in which:
            T11_val = Time_sum_total + Space_sum_total - Mass_sum_total - Lambda_sum_total
            results["T11"] = T11_val
    return results



if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Compute EMT elements between P and Q")
    parser.add_argument('Emax', type=int, help="Maximum energy Emax")
    parser.add_argument('Conf', type=int, help="Number of configurations")
    parser.add_argument('P',    type=int, help="Momentum P")
    parser.add_argument('Q',    type=int, help="Momentum Q")
    parser.add_argument(
        '--eig', '-e',
        dest='Eigen_i',
        type=int,
        default=None,
        help="Eigenvalue index to use. If not specified, 0 for P=Q=0, otherwise 1."
    )

    args = parser.parse_args()

    # 1) Error if exactly one between P and Q is zero
    if (args.P == 0) ^ (args.Q == 0):
        parser.error("Error: P and Q must both be zero or both non-zero.")

    # Extract arguments
    Emax    = args.Emax
    P       = args.P
    Q       = args.Q
    N_conf  = args.Conf
    Eigen_i = args.Eigen_i


    # 2) Smart default value
    if Eigen_i is None:
        Eigen_i = 0 if (P == 0 and Q == 0) else 1

    # 3) Avoid Eigen_i = 0 when P>0 and Q>0
    if P > 0 and Q > 0 and Eigen_i == 0:
        print(
            "[WARN] Eigen_i=0 not valid for P>0 and Q>0; automatically reset to 1.",
            file=sys.stderr
        )
        Eigen_i = 1

    print(f"Doing  Emax={Emax}, P={P}, Q={Q}, Eigen_i={Eigen_i}")

    if P == 0 and Q == 0:
        output_T = compute_T_M0(N_conf, Emax, P, Q, Eigen_i, which=("T00","T01"))
    else:
        output_T = compute_T_PQ(N_conf, Emax, P, Q, Eigen_i, which=("T00", "T11", "T01"), save_mode="yes")
        #compute_Form_Factor(N_conf, Emax, P, Q, Eigen_i, output_T, )
