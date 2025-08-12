from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
import sys
from Htet import *

def particle_parity_classes(states, X):
    """Return the set of parities (even/odd) found across the basis states."""
    parity_classes = set()
    for s, state in enumerate(states):
        parity = n_total(state) % 2
        parity_classes.add(parity)
    return parity_classes

def describe_parity(parity_classes, X):
    """Print a human-readable description of the particle parity for momentum X."""
    if parity_classes == {0}:
        print(f"Momentum: {X} \t Particles: even")
    elif parity_classes == {1}:
        print(f"Momentum: {X} \t Particles: odd")
    else:
        print(f"Momentum: {X} \t Particles: mixed")

def corr3_zero_check(N_particle_p, N_particle_q, P, Q):
    """
    Raise if both sectors are homogeneous and equal (both even or both odd), which makes G^3 vanish.
    Otherwise, print the parity descriptions.
    """
    if (N_particle_p == {0} and N_particle_q == {0}) or (N_particle_p == {1} and N_particle_q == {1}):
        raise ValueError("Error: states contain only particles of the same parity -> G^3 = 0.")
    else:
        describe_parity(N_particle_p, P)
        describe_parity(N_particle_q, Q)

def corr2_zero_check(N_particle_p, N_particle_q, P, Q):
    """
    Raise if both sectors are homogeneous and opposite (even vs odd), which makes G^2 vanish.
    Otherwise, print the parity descriptions.
    """
    if (N_particle_p == {0} and N_particle_q == {1}) or (N_particle_p == {1} and N_particle_q == {0}):
        raise ValueError("Error: states contain only particles of different parity -> G^2 = 0.")
    else:
        describe_parity(N_particle_p, P)
        describe_parity(N_particle_q, Q)

def pad_state_to_python_int(state, delta: int) -> list[int]:
    """Convert entries to Python int and right-pad the occupation list with zeros by 'delta'."""
    converted = [int(x) for x in state]
    if delta <= 0:
        return converted
    return converted + [0] * delta

def load_config_by_index(i, db_path="Data/database.yaml"):
    """Load configuration 'config{i}' from a YAML database and print a compact summary."""
    config_name = f"config{i}"

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at '{db_path}'")

    with open(db_path, 'r') as f:
        config_db = yaml.safe_load(f)

    if config_name not in config_db:
        raise KeyError(f"Configuration '{config_name}' is not present in the database.")

    config = config_db[config_name]
    print(f"Loaded {config_name}: M={config['M']:.2f} Lambda/4π={config['Lambda'] / (4 * math.pi):.2f}, R·2π={config['R'] * 2 * math.pi:.2f}")
    return config

def load_eigenvectors(N_conf, N_eigens, Emax, moments, mode=None, base_dir="Data"):
    """
    Load eigenvectors from a .npz file:
    - N_conf: configuration number (int), e.g. 1 for 'config1'
    - Emax: energy truncation
    - moments: angular (or analogous) momentum used to build the folder name
    - N_eigens: eigenvector index
    - mode: parity ('even', 'odd', or None)
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
    # print(f"Loaded eigenvectors: Momentum = {moments} | Number of states = {len(data['coeffs'])}")
    return {
        "coeffs": data["coeffs"],
        "states": data["states"]
    }

def load_eigenvalues(N_conf, N_eigens, Emax, moments, mode=None, base_dir="Data"):
    """Load (V, VV) eigenvalues and return the indexed entries."""
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
    # print(f"Loaded eigenvalues: Momentum = {moments} | Eigenvalue {N_eigens}")

    return Eigen_V, Eigen_VV

def load_corrections(N_conf, N_eigens, Emax, moments, mode=None, base_dir="Data"):
    """Load counterterm corrections (delta_m^2, delta_lambda^2)."""
    config_folder = f"config{N_conf}"
    moments_str = moments_to_filename(moments)

    folder = os.path.join(base_dir, config_folder, f"Moments_{moments_str}", "Corrections")

    filename = f"Corr_Emax{Emax}.txt"
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")

    delta_m_2, delta_lambda_2 = np.loadtxt(path)
    # print("Loaded corrections")
    return delta_m_2, delta_lambda_2

def lower_k1_k2(Emax, M, R, state, state_index, ix_list, k1, k2):
    """Apply two annihilations a_{k1} a_{k2} on 'state' if possible; return sqrt factors and position."""
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

def raise_k1_k2(Emax, M, R, state, state_index, ix_list, k1, k2):
    """Apply two creations a†_{k1} a†_{k2} with Emax checks; return sqrt factors and position."""
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

def raise_k1_lower_k2(Emax, M, R, state, state_index, ix_list, k1, k2):
    """Apply a†_{k1} a_{k2} when possible with Emax check after the creation."""
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

def raise_k2_lower_k1(Emax, M, R, state, state_index, ix_list, k1, k2):
    """Apply a†_{k2} a_{k1} when possible with Emax check after the creation."""
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

def precompute_2_terms(states1, coeffs1, coeffs2, omega_list1, state_index2, ix_list1, lmax1, Emax1, M, R, term_func):
    """Precompute all two-operator contributions for speed; return list of (prefactor, k)."""
    contributions = []
    for i, state in enumerate(states1):
        c_i = coeffs1[i]
        for k1 in range(-lmax1, lmax1 + 1):
            for k2 in range(-lmax1, lmax1 + 1):
                if omega_list1[k1] <= Emax1 and omega_list1[k2] <= Emax1:
                    fac, pos = term_func(Emax, M, R, state, state_index2, ix_list1, k1, k2)
                    if pos is not None and fac != 0:
                        c_j = coeffs2[pos]
                        prefactor = (
                            c_i * c_j * fac * (1 / (2 * math.pi * R))
                            * (1 / math.sqrt(2 * omega_list1[k1]))
                            * (1 / math.sqrt(2 * omega_list1[k2]))
                        )
                        contributions.append((prefactor, k1))
    return contributions

def efficient_2_sum(R, x, contributions, sign):
    """Efficiently sum Σ prefactor * exp(i * sign * k * x / R)."""
    return sum(prefactor * np.exp(sign * 1j * k * x / R) for prefactor, k in contributions)

def correlation_2_PQ(N_conf, Emax, P, Q, Eigen_i, save_mode='show', save_point_mode='no'):
    """Compute two-point correlator for general momenta P and Q."""
    config = load_config_by_index(N_conf)
    Lambda = config["Lambda"]
    lam_4pi = Lambda / (4 * np.pi)
    M = config["M"]
    R = config["R"]

    p = P / R
    q = Q / R

    Emax_p = np.sqrt((Emax) ** 2 + (p) ** 2)
    Emax_q = np.sqrt((Emax) ** 2 + (q) ** 2)
    Emaxx = max(Emax_p, Emax_q)

    N = 4.0
    if M ** 2 < (Emax_p ** 2) / N:
        lmax_p = int(math.floor(math.sqrt(R ** 2 * (Emax_p ** 2 / N - M ** 2))))
    else:
        lmax_p = 0
    if M ** 2 < (Emax_q ** 2) / N:
        lmax_q = int(math.floor(math.sqrt(R ** 2 * (Emax_q ** 2 / N - M ** 2))))
    else:
        lmax_q = 0

    # Momentum 0 sector for gap extraction
    Evac_0 = load_eigenvalues(N_conf, 0, Emax, moments=[0])[1]
    E1_0 = load_eigenvalues(N_conf, 1, Emax, moments=[0])[1]
    m_gap = E1_0 - Evac_0

    # Momentum P ≠ 0
    E1_p = load_eigenvalues(N_conf, 0, Emax, moments=[P])[1]
    Evac_p = E1_p - np.sqrt(m_gap ** 2 + (p) ** 2)
    Ei_p = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[P])[1]
    results_i_p = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[P])
    coeffs_i_p = results_i_p["coeffs"]
    states_i_p = results_i_p["states"]
    N_particle_p = particle_parity_classes(states_i_p, P)

    # Momentum Q ≠ 0
    E1_q = load_eigenvalues(N_conf, 0, Emax, moments=[Q])[1]
    Evac_q = E1_q - np.sqrt(m_gap ** 2 + (q) ** 2)
    Ei_q = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[Q])[1]
    results_i_q = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[Q])
    coeffs_i_q = results_i_q["coeffs"]
    states_i_q = results_i_q["states"]
    N_particle_q = particle_parity_classes(states_i_q, Q)
    corr2_zero_check(N_particle_p, N_particle_q, P, Q)

    # Harmonise basis size by padding
    lmaxx = max(lmax_p, lmax_q)
    N_common = 2 * lmaxx + 1
    delta_p = N_common - (2 * lmax_p + 1)
    delta_q = N_common - (2 * lmax_q + 1)

    padded_states_i_p = np.array([pad_state_to_python_int(s, delta_p) for s in states_i_p])
    padded_states_i_q = np.array([pad_state_to_python_int(s, delta_q) for s in states_i_q])

    state_index_p = {tuple(ps): i for i, ps in enumerate(padded_states_i_p)}
    state_index_q = {tuple(ps): i for i, ps in enumerate(padded_states_i_q)}

    omega_list_max = {k: omega(k, M, R) for k in range(-lmaxx, lmaxx + 1)}
    ix_list_max = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmaxx, lmaxx + 1)}

    term_sets_2 = {
        'lower_k1_k2': lower_k1_k2,
        'raise_k1_k2': raise_k1_k2,
        'raise_k1_lower_k2': raise_k1_lower_k2,
        'raise_k2_lower_k1': raise_k2_lower_k1,
    }

    # 1) Precomputation with progress bar
    contribs_2 = {}
    print("Precomputing two-term contributions:")
    for name, term in tqdm(term_sets_2.items(), desc="Contrib.2", unit="term"):
        contribs_2[name] = precompute_2_terms(
            padded_states_i_p, coeffs_i_p, coeffs_i_q,
            omega_list_max, state_index_q, ix_list_max,
            lmaxx, Emaxx, M, R, term
        )

    # 2) Evaluation domain
    x_vals = np.linspace(-np.pi * R, np.pi * R, 300)

    # 3) Evaluate f(x) with progress bar
    print("\nEvaluating f(x) on the defined points:")
    f_vals = np.empty_like(x_vals, dtype=complex)
    for idx in tqdm(range(len(x_vals)), desc="Points f(x)", unit="pt"):
        x = x_vals[idx]
        f_vals[idx] = (
            efficient_2_sum(R, x, contribs_2['lower_k1_k2'], sign=-1)
            + efficient_2_sum(R, x, contribs_2['raise_k1_k2'], sign=+1)
            + efficient_2_sum(R, x, contribs_2['raise_k1_lower_k2'], sign=+1)
            + efficient_2_sum(R, x, contribs_2['raise_k2_lower_k1'], sign=-1)
        )

    if save_point_mode == 'yes':
        # Save (x, f(x)) pairs
        save_dir_point = f"Correlation/Points/Corr_2_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i-1}/"
        os.makedirs(save_dir_point, exist_ok=True)
        np.savetxt(os.path.join(save_dir_point, f"La_{lam_4pi:.2f}_R_{R:.2f}.txt"),
                   np.column_stack((x_vals, f_vals)), header="x f(x)")
    else:
        print(f"[INFO] Point saving disabled (save_point_mode='{save_point_mode}')")

    # Split into real/imag parts
    re_vals = np.real(f_vals)
    im_vals = np.imag(f_vals)

    # Plot correlation function
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.size": 28,
        "axes.labelsize": 32,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 26,
        "axes.titlesize": 26
    })

    burnt_orange = '#CC5500'
    uncc_green = '#005035'

    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    ax.plot(x_vals, re_vals, label=r"$\text{Re}[G(x)]$", linewidth=2.8, color=uncc_green)
    ax.plot(x_vals, im_vals, label=r"$\text{Im}[G(x)]$", linestyle="--", color=burnt_orange, linewidth=2.5)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$G(x)$")
    ax.set_xticks([-np.pi * R, 0, np.pi * R])
    ax.set_xticklabels([r"$-\pi R$", r"$0$", r"$\pi R$"])

    lam_4pi = Lambda / (4 * np.pi)
    two_pi_R = int(round(2 * np.pi * R))

    title_str = (
        rf"$E_{{\max}} = {Emax}$, \; "
        rf"$\lambda/4\pi = {lam_4pi:.2f}$, \; "
        rf"$m_{{NO}} = {M:.0f}$, \; "
        rf"$R = {R}$, \; "
        rf"$P = {P}$, \; "
        rf"$Q = {Q}$"
    )
    ax.set_title(title_str, pad=20)

    ax.legend()
    ax.grid(False)
    plt.tight_layout()

    # --- Saving ---
    if save_mode in ('save', 'both'):
        if P != Q:
            save_dir = f"/home/andrea-maestri/University/Tesi_Magistrale/Code/Pictures/Correlation/Corr2_DiffMomenta/Corr_2_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i-1}/"
        else:
            save_dir = f"/home/andrea-maestri/University/Tesi_Magistrale/Code/Pictures/Correlation/Corr2_SameMomenta/Corr_2_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i-1}/"
        os.makedirs(save_dir, exist_ok=True)
        save_fname = f"La_{lam_4pi:.2f}_R_{R:.2f}.png"
        save_path = os.path.join(save_dir, save_fname)

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"[INFO] Figure saved to: {save_path}")

    # --- Show/close ---
    if save_mode in ('show', 'both'):
        plt.show()
    else:
        plt.close()

def correlation_2_00(N_conf, Emax, P, Q, Eigen_i, save_mode='show', save_point_mode='no'):
    """Compute two-point correlator for P=Q=0."""
    config = load_config_by_index(N_conf)
    Lambda = config["Lambda"]
    lam_4pi = Lambda / (4 * np.pi)
    M = config["M"]
    R = config["R"]
    result = load_eigenvectors(N_conf, Eigen_i, Emax, moments=[0])
    coeffs = result["coeffs"]
    states = result["states"]

    N = 4.0
    if M ** 2 < (Emax ** 2) / N:
        lmax = int(math.floor(math.sqrt(R ** 2 * (Emax ** 2 / N - M ** 2))))
    else:
        lmax = 0

    # Build lookup structures
    omega_list = {k: omega(k, M, R) for k in range(-lmax, lmax + 1)}
    ix_list = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmax, lmax + 1)}
    state_index = {tuple(s): i for i, s in enumerate(states)}

    term_sets_2 = {
        'lower_k1_k2': lower_k1_k2,
        'raise_k1_k2': raise_k1_k2,
        'raise_k1_lower_k2': raise_k1_lower_k2,
        'raise_k2_lower_k1': raise_k2_lower_k1,
    }

    # 1) Precompute two-term contributions
    contribs_2 = {}
    print("Precomputing two-term contributions:")
    for name, term in tqdm(term_sets_2.items(), desc="Contrib.2", unit="term"):
        contribs_2[name] = precompute_2_terms(
            states, coeffs, coeffs,
            omega_list, state_index, ix_list,
            lmax, Emax, M, R, term
        )

    # 2) Evaluation domain
    x_vals = np.linspace(-np.pi * R, np.pi * R, 300)

    # 3) Evaluate f(x)
    print("\nEvaluating f(x) on the defined points:")
    f_vals = np.empty_like(x_vals, dtype=complex)
    for idx in tqdm(range(len(x_vals)), desc="Points f(x)", unit="pt"):
        x = x_vals[idx]
        f_vals[idx] = (
            efficient_2_sum(R, x, contribs_2['lower_k1_k2'], sign=-1)
            + efficient_2_sum(R, x, contribs_2['raise_k1_k2'], sign=+1)
            + efficient_2_sum(R, x, contribs_2['raise_k1_lower_k2'], sign=+1)
            + efficient_2_sum(R, x, contribs_2['raise_k2_lower_k1'], sign=-1)
        )

    if save_point_mode == 'yes':
        # Save (x, f(x)) pairs
        save_dir_point = f"Correlation/Points/Corr_2_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i}/"
        os.makedirs(save_dir_point, exist_ok=True)
        np.savetxt(os.path.join(save_dir_point, f"La_{lam_4pi:.2f}_R_{R:.2f}.txt"),
                   np.column_stack((x_vals, f_vals)), header="x f(x)")
    else:
        print(f"[INFO] Point saving disabled (save_point_mode='{save_point_mode}')")

    # Split into real/imag parts
    re_vals = np.real(f_vals)
    im_vals = np.imag(f_vals)

    # Plot correlation function
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.size": 28,
        "axes.labelsize": 32,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 26,
        "axes.titlesize": 26
    })

    burnt_orange = '#CC5500'
    uncc_green = '#005035'

    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    ax.plot(x_vals, re_vals, label=r"$\text{Re}[G(x)]$", linewidth=2.5, color=uncc_green)
    ax.plot(x_vals, im_vals, label=r"$\text{Im}[G(x)]$", linestyle="--", color=burnt_orange, linewidth=2.5)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$G(x)$")
    ax.set_xticks([-np.pi * R, 0, np.pi * R])
    ax.set_xticklabels([r"$-\pi R$", r"$0$", r"$\pi R$"])

    two_pi_R = int(round(2 * np.pi * R))

    title_str = (
        rf"$E_{{\max}} = {Emax}$, \; "
        rf"$\lambda/4\pi = {lam_4pi:.2f}$, \; "
        rf"$m_{{NO}} = {M:.0f}$, \; "
        rf"$R = {R}$, \; "
        rf"$P = {P}$, \; "
        rf"$Q = {Q}$"
    )
    ax.set_title(title_str, pad=20)

    ax.legend()
    ax.grid(False)
    plt.tight_layout()

    # --- Saving ---
    if save_mode in ('save', 'both'):
        save_dir = f"/home/andrea-maestri/University/Tesi_Magistrale/Code/Pictures/Correlation/Corr2_SameMomenta/Corr_2_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i}/"
        os.makedirs(save_dir, exist_ok=True)
        save_fname = f"La_{lam_4pi:.2f}_R_{R:.2f}.png"
        save_path = os.path.join(save_dir, save_fname)

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"[INFO] Figure saved to: {save_path}")

    # --- Show/close ---
    if save_mode in ('show', 'both'):
        plt.show()
    else:
        plt.close()

def raise_k1_raise_k2_raise_k3(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a†_{k1} a†_{k2} a†_{k3} with Emax checks after each creation."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None

    factor1 = math.sqrt(temp_state[ix_k3] + 1)
    temp_state[ix_k3] += 1
    if state_energy(temp_state, M, R) < Emax:
        factor2 = factor1 * math.sqrt(temp_state[ix_k2] + 1)
        temp_state[ix_k2] += 1
        if state_energy(temp_state, M, R) < Emax:
            fac = factor2 * math.sqrt(temp_state[ix_k1] + 1)
            temp_state[ix_k1] += 1
            if state_energy(temp_state, M, R) < Emax:
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def raise_k1_raise_k2_lower_k3(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a†_{k1} a†_{k2} a_{k3} when possible with Emax checks after creations."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None
    if temp_state[ix_k3] != 0:
        factor1 = math.sqrt(temp_state[ix_k3])
        temp_state[ix_k3] -= 1

        factor2 = factor1 * math.sqrt(temp_state[ix_k1] + 1)
        temp_state[ix_k1] += 1
        if state_energy(temp_state, M, R) < Emax:
            fac = factor2 * math.sqrt(temp_state[ix_k2] + 1)
            temp_state[ix_k2] += 1
            if state_energy(temp_state, M, R) < Emax:
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def raise_k3_lower_k1_lower_k2(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a†_{k3} a_{k1} a_{k2} when possible with Emax check after creation."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None
    if temp_state[ix_k2] != 0:
        factor1 = math.sqrt(temp_state[ix_k2])
        temp_state[ix_k2] -= 1
        if temp_state[ix_k1] != 0:
            factor2 = factor1 * math.sqrt(temp_state[ix_k1])
            temp_state[ix_k1] -= 1

            fac = factor2 * math.sqrt(temp_state[ix_k3] + 1)
            temp_state[ix_k3] += 1
            if state_energy(temp_state, M, R) < Emax:
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def lower_k1_lower_k2_lower_k3(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a_{k1} a_{k2} a_{k3} when possible; return factor and position."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None

    if temp_state[ix_k3] != 0:
        factor1 = math.sqrt(temp_state[ix_k3])
        temp_state[ix_k3] -= 1
        if temp_state[ix_k2] != 0:
            factor2 = factor1 * math.sqrt(temp_state[ix_k2])
            temp_state[ix_k2] -= 1
            if temp_state[ix_k1] != 0:
                fac = factor2 * math.sqrt(temp_state[ix_k1])
                temp_state[ix_k1] -= 1
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def raise_k1_raise_k3_lower_k2(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a†_{k1} a†_{k3} a_{k2} when possible with Emax checks after creations."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None

    if temp_state[ix_k2] != 0:
        factor1 = math.sqrt(temp_state[ix_k2])
        temp_state[ix_k2] -= 1

        factor2 = factor1 * math.sqrt(temp_state[ix_k3] + 1)
        temp_state[ix_k3] += 1
        if state_energy(temp_state, M, R) < Emax:
            fac = factor2 * math.sqrt(temp_state[ix_k1] + 1)
            temp_state[ix_k1] += 1
            if state_energy(temp_state, M, R) < Emax:
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def raise_k1_lower_k2_lower_k3(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a†_{k1} a_{k2} a_{k3} when possible with Emax check after creation."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None

    if temp_state[ix_k3] != 0:
        factor1 = math.sqrt(temp_state[ix_k3])
        temp_state[ix_k3] -= 1
        if temp_state[ix_k2] != 0:
            factor2 = factor1 * math.sqrt(temp_state[ix_k2])
            temp_state[ix_k2] -= 1

            fac = factor2 * math.sqrt(temp_state[ix_k1] + 1)
            temp_state[ix_k1] += 1
            if state_energy(temp_state, M, R) < Emax:
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def raise_k2_raise_k3_lower_k1(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a†_{k2} a†_{k3} a_{k1} when possible with Emax checks after creations."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None

    if temp_state[ix_k1] != 0:
        factor1 = math.sqrt(temp_state[ix_k1])
        temp_state[ix_k1] -= 1

        factor2 = factor1 * math.sqrt(temp_state[ix_k3] + 1)
        temp_state[ix_k3] += 1
        if state_energy(temp_state, M, R) < Emax:
            fac = factor2 * math.sqrt(temp_state[ix_k2] + 1)
            temp_state[ix_k2] += 1
            if state_energy(temp_state, M, R) < Emax:
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def raise_k2_lower_k1_lower_k3(Emax, M, R, state, state_index, ix_list, k1, k2, k3):
    """Apply a†_{k2} a_{k1} a_{k3} when possible with Emax check after creation."""
    temp_state = state.copy()
    ix_k1 = ix_list[k1]
    ix_k2 = ix_list[k2]
    ix_k3 = ix_list[k3]

    fac = 0
    pos = None

    if temp_state[ix_k3] != 0:
        factor1 = math.sqrt(temp_state[ix_k3])
        temp_state[ix_k3] -= 1
        if temp_state[ix_k1] != 0:
            factor2 = factor1 * math.sqrt(temp_state[ix_k1])
            temp_state[ix_k1] -= 1

            fac = factor2 * math.sqrt(temp_state[ix_k2] + 1)
            temp_state[ix_k2] += 1
            if state_energy(temp_state, M, R) < Emax:
                pos = state_index.get(tuple(temp_state))
    return fac, pos

def precompute_3_terms(states1, coeffs1, coeffs2, omega_list1, state_index2, ix_list1, lmax1, Emax1, M, R, term_func):
    """Precompute all three-operator contributions; return list of (prefactor, k1, k2)."""
    contributions = []
    for i, state in enumerate(states1):
        c_i = coeffs1[i]
        for k1 in range(-lmax1, lmax1 + 1):
            for k2 in range(-lmax1, lmax1 + 1):
                for k3 in range(-lmax1, lmax1 + 1):
                    fac, pos = term_func(Emax1, M, R, state, state_index2, ix_list1, k1, k2, k3)
                    if pos is not None and fac != 0:
                        c_j = coeffs2[pos]
                        prefactor = (
                            c_i * c_j * fac * 1 / ((2 * math.pi * R) ** (3 / 2))
                            * (1 / math.sqrt(2 * omega_list1[k1]))
                            * (1 / math.sqrt(2 * omega_list1[k2]))
                            * (1 / math.sqrt(2 * omega_list1[k3]))
                        )
                        contributions.append((prefactor, k1, k2))
    return contributions

def efficient_3_sum(R, x, y, contributions, sign1, sign2):
    """Efficiently sum Σ prefactor * exp[i/R (sign1*k1*x + sign2*k2*y)]."""
    phase_factor = 1j / R
    total = 0 + 0j
    for prefactor, k1, k2 in contributions:
        arg = phase_factor * (sign1 * x * k1 + sign2 * y * k2)
        total += prefactor * np.exp(arg)
    return total

def correlation_3_PQ(N_conf, Emax, P, Q, Eigen_i, save_mode='show', save_point_mode='no'):
    """Compute three-point correlator for general momenta P and Q."""
    config = load_config_by_index(N_conf)
    Lambda = config["Lambda"]
    lam_4pi = Lambda / (4 * np.pi)
    M = config["M"]
    R = config["R"]

    p = P / R
    q = Q / R

    Emax_p = np.sqrt((Emax) ** 2 + (p) ** 2)
    Emax_q = np.sqrt((Emax) ** 2 + (q) ** 2)
    Emaxx = max(Emax_p, Emax_q)

    N = 4.0
    if M ** 2 < (Emax_p ** 2) / N:
        lmax_p = int(math.floor(math.sqrt(R ** 2 * (Emax_p ** 2 / N - M ** 2))))
    else:
        lmax_p = 0
    if M ** 2 < (Emax_q ** 2) / N:
        lmax_q = int(math.floor(math.sqrt(R ** 2 * (Emax_q ** 2 / N - M ** 2))))
    else:
        lmax_q = 0

    # Momentum 0 sector for gap extraction
    Evac_0 = load_eigenvalues(N_conf, 0, Emax, moments=[0])[1]
    E1_0 = load_eigenvalues(N_conf, 1, Emax, moments=[0])[1]
    m_gap = E1_0 - Evac_0

    # Momentum P ≠ 0
    E1_p = load_eigenvalues(N_conf, 0, Emax, moments=[P])[1]
    Evac_p = E1_p - np.sqrt(m_gap ** 2 + (p) ** 2)
    Ei_p = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[P])[1]
    results_i_p = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[P])
    coeffs_i_p = results_i_p["coeffs"]
    states_i_p = results_i_p["states"]
    N_particle_p = particle_parity_classes(states_i_p, P)

    # Momentum Q ≠ 0
    E1_q = load_eigenvalues(N_conf, 0, Emax, moments=[Q])[1]
    Evac_q = E1_q - np.sqrt(m_gap ** 2 + (q) ** 2)
    Ei_q = load_eigenvalues(N_conf, Eigen_i - 1, Emax, moments=[Q])[1]
    results_i_q = load_eigenvectors(N_conf, Eigen_i - 1, Emax, moments=[Q])
    coeffs_i_q = results_i_q["coeffs"]
    states_i_q = results_i_q["states"]
    N_particle_q = particle_parity_classes(states_i_q, Q)

    corr3_zero_check(N_particle_p, N_particle_q, P, Q)

    # Harmonise basis size by padding
    lmaxx = max(lmax_p, lmax_q)
    N_common = 2 * lmaxx + 1
    delta_p = N_common - (2 * lmax_p + 1)
    delta_q = N_common - (2 * lmax_q + 1)

    padded_states_i_p = np.array([pad_state_to_python_int(s, delta_p) for s in states_i_p])
    padded_states_i_q = np.array([pad_state_to_python_int(s, delta_q) for s in states_i_q])

    state_index_p = {tuple(ps): i for i, ps in enumerate(padded_states_i_p)}
    state_index_q = {tuple(ps): i for i, ps in enumerate(padded_states_i_q)}

    omega_list_max = {k: omega(k, M, R) for k in range(-lmaxx, lmaxx + 1)}
    ix_list_max = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmaxx, lmaxx + 1)}

    term_sets = {
        'raise_k1_raise_k2_raise_k3': raise_k1_raise_k2_raise_k3,
        'raise_k1_raise_k2_lower_k3': raise_k1_raise_k2_lower_k3,
        'raise_k3_lower_k1_lower_k2': raise_k3_lower_k1_lower_k2,
        'lower_k1_lower_k2_lower_k3': lower_k1_lower_k2_lower_k3,
        'raise_k1_raise_k3_lower_k2': raise_k1_raise_k3_lower_k2,
        'raise_k1_lower_k2_lower_k3': raise_k1_lower_k2_lower_k3,
        'raise_k2_raise_k3_lower_k1': raise_k2_raise_k3_lower_k1,
        'raise_k2_lower_k1_lower_k3': raise_k2_lower_k1_lower_k3,
    }

    # Precompute contributions
    contribs = {}
    print("Precomputing contributions:")
    for name, term in tqdm(term_sets.items(), desc="Contributions", unit="term"):
        contribs[name] = precompute_3_terms(
            padded_states_i_p, coeffs_i_p, coeffs_i_q,
            omega_list_max, state_index_q, ix_list_max,
            lmaxx, Emaxx, M, R, term
        )

    # Grids for f(x, y)
    n_x, n_y = 100, 100
    x_vals = np.linspace(-np.pi * R, np.pi * R, n_x)
    y_vals = np.linspace(-np.pi * R, np.pi * R, n_y)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    # Accumulate values
    f_vals = np.zeros_like(X, dtype=complex)

    print("\nEvaluating G^3 on the grid:")
    total_points = n_x * n_y
    for idx in tqdm(range(total_points), desc="Points", unit="pt"):
        i = idx // n_y
        j = idx % n_y
        x, y = X[i, j], Y[i, j]

        f_vals[i, j] = (
            efficient_3_sum(R, x, y, contribs['raise_k1_raise_k2_raise_k3'], sign1=+1, sign2=+1)
            + efficient_3_sum(R, x, y, contribs['raise_k1_raise_k2_lower_k3'], sign1=+1, sign2=+1)
            + efficient_3_sum(R, x, y, contribs['raise_k3_lower_k1_lower_k2'], sign1=-1, sign2=-1)
            + efficient_3_sum(R, x, y, contribs['lower_k1_lower_k2_lower_k3'], sign1=-1, sign2=-1)
            + efficient_3_sum(R, x, y, contribs['raise_k1_raise_k3_lower_k2'], sign1=+1, sign2=-1)
            + efficient_3_sum(R, x, y, contribs['raise_k1_lower_k2_lower_k3'], sign1=+1, sign2=-1)
            + efficient_3_sum(R, x, y, contribs['raise_k2_raise_k3_lower_k1'], sign1=-1, sign2=+1)
            + efficient_3_sum(R, x, y, contribs['raise_k2_lower_k1_lower_k3'], sign1=-1, sign2=+1)
        )

    re_vals = np.real(f_vals)
    im_vals = np.imag(f_vals)

    if save_point_mode == 'yes':
        # Save (x, y, Re f, Im f)
        save_dir_point = f"Correlation/Points/Corr_3_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i-1}/"
        os.makedirs(save_dir_point, exist_ok=True)

        data = np.column_stack((
            X.flatten(),
            Y.flatten(),
            f_vals.real.flatten(),
            f_vals.imag.flatten()
        ))

        header = "x y Re[f(x,y)] Im[f(x,y)]"
        file_path = os.path.join(save_dir_point, f"La_{lam_4pi:.2f}_R_{R:.2f}.txt")
        np.savetxt(file_path, data, header=header)
    else:
        print(f"[INFO] Point saving disabled (save_point_mode='{save_point_mode}')")

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.size": 28,
        "axes.labelsize": 32,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 26,
        "axes.titlesize": 26
    })

    burnt_orange = '#CC5500'
    uncc_green = '#005035'

    # 3D plot of the real part
    fig = plt.figure(figsize=(16, 9), )
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    surf = ax.plot_surface(X, Y, re_vals, cmap='jet', edgecolor='black', linewidth=0.25, antialiased=True)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$\text{Re}[G(x,y)]$")

    ax.set_xticks([-np.pi * R, 0, np.pi * R])
    ax.set_xticklabels([r"$-\pi R$", r"$0$", r"$\pi R$"])

    ax.set_yticks([-np.pi * R, 0, np.pi * R])
    ax.set_yticklabels([r"$-\pi R$", r"$0$", r"$\pi R$"])

    lam_4pi = Lambda / (4 * np.pi)
    lam_4pi_str = f"{lam_4pi:.2f}"
    R_str = f"{R:.2f}"
    title_str = (
        rf"$E_{{\max}} = {Emax}$, \; "
        rf"$\lambda/4\pi = {lam_4pi:.2f}$, \; "
        rf"$m_{{NO}} = {M:.0f}$, \; "
        rf"$R = {R}$, \; "
        rf"$P = {P}$, \; "
        rf"$Q = {Q}$"
    )
    ax.set_title(title_str, pad=20)
    ax.set_xlabel(r"$x$", labelpad=20)
    ax.set_ylabel(r"$y$", labelpad=20)
    ax.set_zlabel(r"$\mathrm{Re}[G(x,y)]$", labelpad=20)
    plt.tight_layout()

    # --- Saving ---
    if save_mode in ('save', 'both'):
        if P != Q:
            save_dir = f"/home/andrea-maestri/University/Tesi_Magistrale/Code/Pictures/Correlation/Corr3_DiffMomenta/Corr_3_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i-1}/"
        else:
            save_dir = f"/home/andrea-maestri/University/Tesi_Magistrale/Code/Pictures/Correlation/Corr3_SameMomenta/Corr_3_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i-1}/"
        os.makedirs(save_dir, exist_ok=True)
        save_fname = f"La_{lam_4pi:.2f}_R_{R:.2f}.png"
        save_path = os.path.join(save_dir, save_fname)

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"[INFO] Figure saved to: {save_path}")

    # --- Show/close ---
    if save_mode in ('show', 'both'):
        plt.show()
    else:
        plt.close()

def correlation_3_00(N_conf, Emax, P, Q, Eigen_i, save_mode='show', save_point_mode='no'):
    """Three-point correlator with P=Q=0 is forbidden when both sectors have the same homogeneous parity."""
    raise ValueError("Error: both states contain only particles with the same parity (both even or both odd).")
    # (Dead code below retained intentionally to preserve structure; function exits via raise.)
    config = load_config_by_index(N_conf)
    Lambda = config["Lambda"]
    lam_4pi = Lambda / (4 * np.pi)
    M = config["M"]
    R = config["R"]
    result = load_eigenvectors(N_conf, Eigen_i, Emax, moments=[1])
    coeffs = result["coeffs"]
    states = result["states"]

    N = 4.0
    if M ** 2 < (Emax ** 2) / N:
        lmax = int(math.floor(math.sqrt(R ** 2 * (Emax ** 2 / N - M ** 2))))
    else:
        lmax = 0

    omega_list = {k: omega(k, M, R) for k in range(-lmax, lmax + 1)}
    ix_list = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmax, lmax + 1)}
    state_index = {tuple(s): i for i, s in enumerate(states)}

    contrib_raise_k1_k2_k3 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, raise_k1_raise_k2_raise_k3)
    contrib_raise_k1_k2_lower_k3 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, raise_k1_raise_k2_lower_k3)
    contrib_raise_k3_lower_k1_lower_k2 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, raise_k3_lower_k1_lower_k2)
    contrib_lower_k1_k2_k3 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, lower_k1_lower_k2_lower_k3)
    contrib_raise_k1_raise_k3_lower_k2 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, raise_k1_raise_k3_lower_k2)
    contrib_raise_k1_lower_k2_lower_k3 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, raise_k1_lower_k2_lower_k3)
    contrib_raise_k2_raise_k3_lower_k1 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, raise_k2_raise_k3_lower_k1)
    contrib_raise_k2_lower_k1_lower_k3 = precompute_3_terms(states, coeffs, coeffs, omega_list, state_index, ix_list, lmax, Emax, M, R, raise_k2_lower_k1_lower_k3)

    n_x, n_y = 100, 100
    x_vals = np.linspace(-np.pi * R, np.pi * R, n_x)
    y_vals = np.linspace(-np.pi * R, np.pi * R, n_y)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    f_vals = np.zeros_like(X, dtype=complex)
    for i in range(n_x):
        for j in range(n_y):
            x, y = X[i, j], Y[i, j]
            f_vals[i, j] = (
                efficient_3_sum(R, x, y, contrib_raise_k1_k2_k3, sign1=+1, sign2=+1)
                + efficient_3_sum(R, x, y, contrib_raise_k1_k2_lower_k3, sign1=+1, sign2=+1)
                + efficient_3_sum(R, x, y, contrib_raise_k3_lower_k1_lower_k2, sign1=-1, sign2=-1)
                + efficient_3_sum(R, x, y, contrib_lower_k1_k2_k3, sign1=-1, sign2=-1)
                + efficient_3_sum(R, x, y, contrib_raise_k1_raise_k3_lower_k2, sign1=+1, sign2=-1)
                + efficient_3_sum(R, x, y, contrib_raise_k1_lower_k2_lower_k3, sign1=+1, sign2=-1)
                + efficient_3_sum(R, x, y, contrib_raise_k2_raise_k3_lower_k1, sign1=-1, sign2=+1)
                + efficient_3_sum(R, x, y, contrib_raise_k2_lower_k1_lower_k3, sign1=-1, sign2=+1)
            )

    re_vals = np.real(f_vals)
    im_vals = np.imag(f_vals)

    if save_point_mode == 'yes':
        save_dir_point = f"Correlation/Points/Corr_3_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i}/"
        os.makedirs(save_dir_point, exist_ok=True)

        data = np.column_stack((
            X.flatten(),
            Y.flatten(),
            f_vals.real.flatten(),
            f_vals.imag.flatten()
        ))

        header = "x y Re[f(x,y)] Im[f(x,y)]"
        file_path = os.path.join(save_dir_point, f"La_{lam_4pi:.2f}_R_{R:.2f}.txt")
        np.savetxt(file_path, data, header=header)
    else:
        print(f"[INFO] Point saving disabled (save_point_mode='{save_point_mode}')")

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.size": 28,
        "axes.labelsize": 32,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 26,
        "axes.titlesize": 26
    })

    burnt_orange = '#CC5500'
    uncc_green = '#005035'

    fig = plt.figure(figsize=(16, 9), )
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    surf = ax.plot_surface(X, Y, re_vals, cmap='jet', edgecolor='black', linewidth=0.25, antialiased=True)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$\text{Re}[G(x,y)]$")

    ax.set_xticks([-np.pi * R, 0, np.pi * R])
    ax.set_xticklabels([r"$-\pi R$", r"$0$", r"$\pi R$"])

    ax.set_yticks([-np.pi * R, 0, np.pi * R])
    ax.set_yticklabels([r"$-\pi R$", r"$0$", r"$\pi R$"])

    lam_4pi = Lambda / (4 * np.pi)
    lam_4pi_str = f"{lam_4pi:.2f}"
    R_str = f"{R:.2f}"
    title_str = (
        rf"$E_{{\max}} = {Emax}$, \; "
        rf"$\lambda/4\pi = {lam_4pi:.2f}$, \; "
        rf"$m_{{NO}} = {M:.0f}$, \; "
        rf"$R = {R}$, \; "
        rf"$P = {P}$, \; "
        rf"$Q = {Q}$"
    )
    ax.set_title(title_str, pad=20)
    ax.set_xlabel(r"$x$", labelpad=20)
    ax.set_ylabel(r"$y$", labelpad=20)
    ax.set_zlabel(r"$\text{Re}[G(x,y)]$", labelpad=20)
    plt.tight_layout()

    if save_mode in ('save', 'both'):
        save_dir = f"/home/andrea-maestri/University/Tesi_Magistrale/Code/Pictures/Correlation/Corr3_SameMomenta/Corr_3_P{P}_Q{Q}/E{Emax:.2f}/Eigen_{Eigen_i-1}/"
        os.makedirs(save_dir, exist_ok=True)
        save_fname = f"La_{lam_4pi:.2f}_R_{R:.2f}.png"
        save_path = os.path.join(save_dir, save_fname)

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"[INFO] Figure saved to: {save_path}")

    if save_mode in ('show', 'both'):
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Compute correlation function for momenta P and Q"
    )
    parser.add_argument('--version', '-v', choices=['two', 'three', 'both'],
                        help="Which version to compute: 'two', 'three', or 'both'. Default: 'two'.")
    parser.add_argument('Emax', type=int, help="Maximum energy Emax")
    parser.add_argument('Conf', type=int, help="Configuration number")
    parser.add_argument('P',    type=int, help="Momentum P")
    parser.add_argument('Q',    type=int, help="Momentum Q")
    parser.add_argument(
        '--eig', '-e',
        dest='Eigen_i',
        type=int,
        default=None,
        help="Eigenvalue index to use. If not specified: 0 for P=Q=0, otherwise 1."
    )

    args = parser.parse_args()

    # 1) Error if exactly one between P and Q is zero
    if (args.P == 0) ^ (args.Q == 0):
        parser.error("Error: P and Q must be both zero or both non-zero.")

    # Extract arguments
    Emax = args.Emax
    P = args.P
    Q = args.Q
    N_conf = args.Conf
    Eigen_i = args.Eigen_i

    # 2) Smart default for Eigen_i
    if Eigen_i is None:
        Eigen_i = 0 if (P == 0 and Q == 0) else 1

    # 3) Avoid Eigen_i = 0 when P > 0 and Q > 0
    if P > 0 and Q > 0 and Eigen_i == 0:
        print(
            "[WARN] Eigen_i=0 is invalid for P>0 and Q>0; automatically reset to 1.",
            file=sys.stderr
        )
        Eigen_i = 1

    print(f"\nDoing {args.version} with Emax={Emax}, P={P}, Q={Q}, Eigen_i={Eigen_i}")

    if P == 0 and Q == 0:

        if args.version in ['two', 'both']:
            correlation_2_00(N_conf, Emax, P, Q, Eigen_i, save_mode='save', save_point_mode='yes')
        if args.version in ['three', 'both']:
            correlation_3_00(N_conf, Emax, P, Q, Eigen_i, save_mode='save', save_point_mode='yes')
    else:
        if args.version in ['two', 'both']:
            correlation_2_PQ(N_conf, Emax, P, Q, Eigen_i, save_mode='save', save_point_mode='yes')
        if args.version in ['three', 'both']:
            correlation_3_PQ(N_conf, Emax, P, Q, Eigen_i, save_mode='save', save_point_mode='yes')
