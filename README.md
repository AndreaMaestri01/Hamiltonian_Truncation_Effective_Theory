# Hamiltonian Truncation Effective Theory for a Scalar $\phi^4$ Theory on $S^1$

This repository contains three Python modules that implement a compact workflow for Hamiltonian Truncation Effective Theory (HTET) in $1+1$ dimensions on a compact spatial circle. The code:

1. builds a Fock basis with fixed total momentum and (optionally) fixed $\mathbb{Z}_2$ parity,
2. assembles the effective Hamiltonians $H_{\text{eff}}^{(V)}$ and $lambdaH_{\text{eff}}^{(V^2)}$, computes the lowest eigenpairs, and stores them to disk,
3. evaluates matrix elements of the energy–momentum tensor $T_{\mu\nu}$ between finite–momentum eigenstates, and
4. computes two– and three–point correlation functions from the stored eigenvectors.

The codebase is organised as follows:

* `Htet.py`: basis generation, operator construction $(H_0, H_2, H_4)$, effective Hamiltonians, eigenpairs, and persistent storage.
* `EMT.py`: form–factor–like matrix elements of $T_{\mu\nu}$ between eigenstates of momenta $P$ and $Q$, with options to save tabulated results.
* `CorrelationFunction.py`: position–space correlators $G^2(x)$ and $G^3(x,y)$, including parity-based zero checks, efficient precomputation, plotting, and optional point dumps.

---

## 1. Prerequisites

* Python 3.10+ (recommended).
* Packages: `numpy`, `scipy` (including `scipy.sparse`), `matplotlib`, `pyyaml`, `tqdm`.

---

## 2. Physics and conventions (concise)

* Spatial manifold: a circle of radius $R$. Single–particle frequencies are $\omega_k=\sqrt{(k/R)^2+M^2}$. The integer label $k=\ell$ is the (discrete) angular momentum / Fourier mode.
* Basis: Fock states $|n_0,n_1,n_2,\dots\rangle$, with a fixed set of allowed momenta and an energy truncation $E_{\max}$. The total free energy of a state is $\sum_i n_i\,\omega_{\ell(i)}$.
* $\mathbb{Z}_2$ parity is the total number of quanta mod 2; even/odd sectors may be selected.
* Operators:

  * $H_0$: diagonal matrix of free energies.
  * $H_2$: normal–ordered $\int dx\,:\!\phi^2\!:$ split into $(\phi^-)^2+(\phi^+)^2+2\phi^-\phi^+$.
  * $H_4$: normal–ordered $\int dx\,:\!\phi^4\!:$ built from all combinatoric channels $(\phi^\pm)^m(\phi^\mp)^{4-m}$ with momentum conservation.
* Effective Hamiltonians:
  $H_{\text{eff}}^{(V)}=H_0+\frac{\lambda}{24}\,H_4$, and an $O(V^2)$–improved version $lambdaH_{\text{eff}}^{(V^2)}=H_0+\frac{m_2}{2}\,H_2+\frac{\lambda+\lambda_2}{24}\,H_4$, with $m_2,\lambda_2$ computed from UV sums.

---

## 3. Data layout and persistence

A lightweight database `Data/database.yaml` stores **configurations** as named entries `configX` with the triple $\{M,\lambda,R\}$. Eigenvectors, eigenvalues, counterterm corrections, and timing information are stored under a configuration–aware hierarchy:

```
Data/
  database.yaml
  configX/
    Moments_<tag>/
      Eigenvectors[_even|_odd]/  # Eigenvec_Emax{Emax}_n{n}.npz
      Eigenvalues[_even|_odd]/   # Eigen_Emax{Emax}_V.txt, Eigen_Emax{Emax}_VV.txt
      Corrections/               # Corr_Emax{Emax}.txt
      Computation/               # computation_cost.txt
```

The `<tag>` encodes the list of allowed momenta via `moments_to_filename`, e.g. `l0_l1_lm1`.

**Filenames and meaning.**

* `Eigenvec_Emax{Emax}_n{n}.npz`: arrays `coeffs` and `states` for the $n$-th eigenvector.
* `Eigen_Emax{Emax}_V.txt`, `Eigen_Emax{Emax}_VV.txt`: lowest eigenvalues with and without $O(V^2)$ improvements.
* `Corr_Emax{Emax}.txt`: the pair $(\delta m^2,\delta\lambda)$ used in $lambdaH_{\text{eff}}^{(V^2)}$.
* `computation_cost.txt`: rows `Emax Nstate TimeUsed_secondes`, one per new $E_{\max}$.

---

## 4. Typical workflow

### Step 1 — Generate eigenpairs

Run `Htet.py` from the command line:

```bash
python Htet.py Emax Moments Ray La [mode]
```

* `Emax` (int): free–theory energy truncation (if total momentum $N_{\text{tot}}\neq 0$, the code uses $\sqrt{E_{\max}^2+(N_{\text{tot}}/R)^2}$).
* `Moments` (str): comma–separated list of allowed total momenta; the current CLI expects **exactly one** integer, e.g. `0` or `1`.
* `Ray` (float): scale for the **effective** radius via $R=(10/(2\pi))\times \text{Ray}$.
* `La` (float): coupling as $\lambda=4\pi\,\text{La}$.
* `[mode]` (optional): choose `'even'` or `'odd'` to restrict to a fixed $\mathbb{Z}_2$ parity sector.

On success, the script builds the basis, assembles $H_0,H_2,H_4$, constructs $H_{\text{eff}}^{(V)}$ and $lambdaH_{\text{eff}}^{(V^2)}$, computes the lowest eigenpairs, and saves eigenvalues/eigenvectors and $(\delta m^2,\delta\lambda)$ to the `Data/` tree.

> **Note.** Internally the code uses: basis generators and filters; the map `the_l(ix)` that interleaves $+\ell$ and $-\ell$; $\omega(k,M,R)$; and sparse builders for the various normal–ordered pieces.

### Step 2 — Compute $T_{\mu\nu}$ matrix elements

Use the functions in `EMT.py` once eigenpairs are present in `Data/`:

* Load configuration, eigenvalues, eigenvectors, and counterterms via the provided helpers.
* Call `compute_T_M0(N_conf,Emax,P,Q,Eigen_i,which=("T01","T00","T11"))` to obtain a dictionary with any subset of $\{T_{01},T_{00},T_{11}\}$ between momentum–$P$ and momentum–$Q$ eigenstates (index `Eigen_i` in the saved ordering). The function supports saving rows to `EMT/Points/config{N_conf}/Emax_{Emax}/FormFactor.txt` (toggle with `save_mode="yes"` inside the function).
* Internally, $T_{01}$ is assembled from four two–operator components; $T_{00}$ and $T_{11}$ are built from “mass”, “time”, “space”, and “$\lambda$” sums, with optional accuracy diagnostics when $P=Q$.

### Step 3 — Correlation functions

`CorrelationFunction.py` computes:

* $G^2(x)$: `correlation_2_PQ(N_conf,Emax,P,Q,Eigen_i,save_mode,save_point_mode)`. It harmonises bases across $P$ and $Q$, precomputes all two–operator terms, evaluates the correlator on a grid in $x$, and produces a LaTeX–styled plot; optionally saves $(x,f(x))$ to `Correlation/Points/...`.
* $G^3(x,y)$: analogous machinery in two variables, with progress bars and optional point dumps.

**Parity-based zero checks.**

* $G^2$ vanishes if both sectors are homogeneous and of **opposite** parity (even vs odd). The code detects this and raises a descriptive error.
* $G^3$ vanishes if both sectors are homogeneous and of the **same** parity. A guard raises accordingly.

---

## 5. Minimal examples (Python REPL)

> These examples assume you already populated `Data/` by running `Htet.py` at least once with your parameters.

```python
# Example: read eigenpairs and compute EMT elements
from EMT import compute_T_M0
res = compute_T_M0(N_conf=1, Emax=8, P=1, Q=1, Eigen_i=1, which=("T01","T00","T11"))
print(res)  # {'T01': ..., 'T00': ..., 'T11': ...}
```

This uses the readers `load_config_by_index`, `load_eigenvalues`, `load_eigenvectors`, and counterterms `load_correction` under the hood.

```python
# Example: two-point correlator
from CorrelationFunction import correlation_2_PQ
correlation_2_PQ(N_conf=1, Emax=8, P=1, Q=1, Eigen_i=1, save_mode='show', save_point_mode='no')
```

This performs precomputation of two–operator contributions and then evaluates/plots $G^2(x)$.

---

## 6. Important implementation details

* **Basis and indices.** The mapping `the_l(ix)` orders $(+\ell,-\ell)$ pairs (with a special entry at $\ell=0$); helper functions compute $n_{\text{tot}}$, $l_{\text{tot}}$, and the free energy of each state.
* **Sparse assembly.** Each normal–ordered channel (e.g. $(\phi^+)^2$, $\phi^-\phi^+$, $(\phi^+)^4$, $\phi^-(\phi^+)^3$, $(\phi^-)^2(\phi^+)^2$) has enumerators that enforce momentum conservation and energy–cutoff constraints before inserting nonzeros.
* **Parity restriction.** When saving/reading, parity suffixes `_even` and `_odd` are appended to `Eigenvectors/` and `Eigenvalues/` as appropriate. Readers in `EMT.py` and `CorrelationFunction.py` respect this convention.
* **Configuration database.** On first save, a new `configX` is created or an existing one is reused by matching the triple $(\lambda,M,R)$. Subsequent data for the same configuration are stored beneath the corresponding folder.
* **Timings.** The function `save_time_number` collects the basis size and wall time per $E_{\max}$.
* **Plotting.** Correlator plots use LaTeX rendering and custom fonts; some output paths are *hard–coded* to an absolute user directory. Adjust these paths before batch runs on other machines.

---

## 7. Reproducibility tips

* Always record the tuple $(\lambda/4\pi,M,R,E_{\max},\text{Moments},\text{mode})$. The saved file names and directory structure encode most of this information; `database.yaml` holds the rest.
* When comparing $T_{00}$ at $P=Q$ with the corresponding eigenvalue, the code prints a relative accuracy diagnostic; keep an eye on it when varying $E_{\max}$.
* Use consistent `Moments` tags across runs; the same tag must be used to read back the correct eigenpairs.

---

## 8. Troubleshooting

* **Missing files** during read–back. The loaders raise explicit errors if a requested file is absent; check `N_conf`, `Emax`, `Moments` tag, and parity `mode`.
* **Vanishing correlators.** If you hit a `ValueError` about parity in $G^2$ or $G^3$, it is by design (selection rules). Choose compatible sectors.
* **Absolute save paths** (plots, EMT points). Some routines save under absolute paths tailored to the developer’s environment; modify them for your setup.

---

## 9. Citation and authorship

The main script attributes authorship and acknowledges adapted portions from an external Hamiltonian–truncation repository; please retain these notices if you redistribute.

---

## 10. Quick reference (APIs)

* **Htet.py**:
  `gen_basis`, `filter_even/odd`, `filter_moments`, `state_energy`, `l_total`, `n_total`, `H0`, `H2`, `H4`, `Matrices`, `Eigens`, `save_eigenvalues`, `save_eigenvectors`, `save_correction`, `save_time_number`, `moments_to_filename`. See in–code docstrings and builders for details.
* **EMT.py**:
  `compute_T_M0`, `compute_T01_terms`, low/raise operator kernels, and readers `load_config_by_index`, `load_eigenvalues`, `load_eigenvectors`, `load_correction`.
* **CorrelationFunction.py**:
  `correlation_2_PQ`, `correlation_2_00`, `correlation_3_PQ`, `correlation_3_00`, with helper parity checks and efficient precomputation/evaluation routines.

---

### A final note

The codebase is intentionally modular and explicit. It prioritises clarity in the construction of operators and careful bookkeeping of configuration metadata. For large $E_{\max}$, the *precomputation* strategy in the correlator routines and the *sparse* representations in the Hamiltonian assembly are essential for tractable performance on a workstation.

