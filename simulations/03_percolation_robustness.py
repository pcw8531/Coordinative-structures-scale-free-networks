#!/usr/bin/env python3
"""
03_percolation_robustness.py
============================
Manuscript Sections 2.3 (Correspondences 1–3) & 3.3 — Percolation and Robustness

Implements two complementary analyses:
  A. Bond percolation sweeps (Eqs 6, 9, 10)
  B. Robustness-fragility analysis (Eq 7)

Part A — Percolation Dynamics:
  - Order parameter: P_∞(p) = |C₁(p)| / N  (Eq 9)
  - Susceptibility:  χ(p) = dP_∞ / dp  (Eq 10)
  - Percolation threshold: p_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩) (Eq 6)
  Sweep p ∈ [0, 1] in steps of 0.01, 100 realizations.

Part B — Robustness-Fragility:
  - Fragmentation ratio: f_r(q) = 1 − S(q) / S(0)  (Eq 7)
  - Asymmetry ratio: A_r(q) = f_r(targeted, q) / f_r(random, q)
  Removal fractions q ∈ {0.01, ..., 0.50}, 50 random trials.

Target outputs: Tables S4, S5, S5b, and Table 2 (percolation rows).

Parameters (Table S6b):
  Bond occupation sweep: p ∈ [0, 1], step 0.01
  P_∞ threshold: 0.5
  Network realizations: 100

Output:
  data/simulation_outputs/percolation_results.csv
  data/simulation_outputs/robustness_results.csv
  data/simulation_outputs/percolation_ensemble_data.npz

Usage:
  python simulations/03_percolation_robustness.py
"""

import numpy as np
import networkx as nx
import os
import time

# ══════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════
N = 100
BASE_SEED = 42
N_REALIZATIONS = 100

# Network parameters
ER_P = 6.0 / (N - 1)
WS_K = 6
WS_BETA = 0.1
BA_M = 3

# Percolation sweep parameters (Table S6b)
P_RANGE = np.arange(0.0, 1.01, 0.01)    # Bond occupation p ∈ [0, 1]
P_INF_THRESHOLD = 0.5                     # Giant component threshold

# Robustness parameters (S3 Appendix Part C)
Q_VALUES = np.array([0.01, 0.02, 0.03, 0.04, 0.05,
                     0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.40, 0.50])
N_RANDOM_TRIALS = 50   # Random removal trials per q value
Q_REPORT = 0.28        # Primary reporting threshold for A_r

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'data', 'simulation_outputs')


# ══════════════════════════════════════════════════════════════
# NETWORK GENERATORS
# ══════════════════════════════════════════════════════════════
def generate_network(topo, seed):
    if topo == 'ER':
        return nx.erdos_renyi_graph(N, ER_P, seed=seed)
    elif topo == 'WS':
        return nx.watts_strogatz_graph(N, WS_K, WS_BETA, seed=seed)
    elif topo == 'BA':
        return nx.barabasi_albert_graph(N, BA_M, seed=seed)


# ══════════════════════════════════════════════════════════════
# PART A: BOND PERCOLATION — Eqs 6, 9, 10
# ══════════════════════════════════════════════════════════════
def bond_percolation_sweep(G, p_range, rng):
    """
    Perform bond percolation sweep over the range of occupation probabilities.

    For each p, independently retain each edge with probability p,
    then compute the giant component fraction P_∞(p) = |C₁(p)| / N (Eq 9).

    Returns:
        p_inf: array of P_∞ values for each p
    """
    edges = list(G.edges())
    n_edges = len(edges)
    n_nodes = G.number_of_nodes()

    p_inf = np.zeros(len(p_range))

    for i, p in enumerate(p_range):
        if p == 0:
            p_inf[i] = 1.0 / n_nodes  # Each node is its own component
            continue
        if p >= 1.0:
            if nx.is_connected(G):
                p_inf[i] = 1.0
            else:
                gcc = max(nx.connected_components(G), key=len)
                p_inf[i] = len(gcc) / n_nodes
            continue

        # Retain each edge with probability p
        mask = rng.random(n_edges) < p
        retained = [edges[j] for j in range(n_edges) if mask[j]]

        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(retained)

        if H.number_of_edges() == 0:
            p_inf[i] = 1.0 / n_nodes
        else:
            gcc = max(nx.connected_components(H), key=len)
            p_inf[i] = len(gcc) / n_nodes

    return p_inf


def compute_susceptibility(p_range, p_inf):
    """
    Eq 10: χ(p) = dP_∞ / dp

    Compute numerical derivative using central differences.
    """
    chi = np.gradient(p_inf, p_range)
    return chi


def find_percolation_threshold(p_range, p_inf, threshold=0.5):
    """Find p_c where P_∞ first exceeds threshold."""
    above = np.where(p_inf >= threshold)[0]
    if len(above) == 0:
        return p_range[-1]
    return p_range[above[0]]


def theoretical_pc(G):
    """
    Eq 6: p_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩)

    Molloy–Reed criterion for uncorrelated networks.
    """
    degrees = np.array([G.degree(n) for n in G.nodes()])
    k_mean = degrees.mean()
    k2_mean = (degrees ** 2).mean()
    kappa = k2_mean / k_mean
    if kappa <= 1:
        return 1.0
    return 1.0 / (kappa - 1)


# ══════════════════════════════════════════════════════════════
# PART B: ROBUSTNESS-FRAGILITY — Eq 7
# ══════════════════════════════════════════════════════════════
def largest_component_size(G):
    """Return size of the largest connected component."""
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


def robustness_random_removal(G, q_values, n_trials, rng):
    """
    Random node removal: for each q, remove ⌊q·N⌋ nodes randomly,
    compute S(q), average over n_trials.

    Returns:
        f_r_random: fragmentation ratios f_r(q) = 1 − S(q)/S(0) (Eq 7)
    """
    s0 = largest_component_size(G)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    f_r = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        n_remove = max(1, int(q * n_nodes))
        s_trials = []
        for _ in range(n_trials):
            to_remove = rng.choice(nodes, size=n_remove, replace=False)
            H = G.copy()
            H.remove_nodes_from(to_remove)
            s_trials.append(largest_component_size(H))
        f_r[i] = 1.0 - np.mean(s_trials) / s0

    return f_r


def robustness_targeted_removal(G, q_values):
    """
    Targeted hub removal: remove nodes in decreasing degree order.

    Returns:
        f_r_targeted: fragmentation ratios f_r(q) = 1 − S(q)/S(0)
    """
    s0 = largest_component_size(G)
    nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    n_nodes = len(nodes)
    f_r = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        n_remove = max(1, int(q * n_nodes))
        H = G.copy()
        H.remove_nodes_from(nodes[:n_remove])
        s_q = largest_component_size(H)
        f_r[i] = 1.0 - s_q / s0

    return f_r


# ══════════════════════════════════════════════════════════════
# MAIN ENSEMBLE SIMULATION
# ══════════════════════════════════════════════════════════════
def run_ensemble():
    print("=" * 70)
    print("03_percolation_robustness.py")
    print("Percolation Dynamics (Eqs 6, 9, 10) & Robustness (Eq 7)")
    print(f"N = {N}, {N_REALIZATIONS} realizations, seed = {BASE_SEED}")
    print("=" * 70)

    topologies = ['ER', 'WS', 'BA']
    results = {}
    ensemble_data = {}

    for topo in topologies:
        print(f"\n--- {topo} ---")
        t0 = time.time()

        all_p_inf = np.zeros((N_REALIZATIONS, len(P_RANGE)))
        all_pc_sim = []
        all_pc_theo = []
        all_p_peak = []
        all_chi_max = []

        all_fr_random = np.zeros((N_REALIZATIONS, len(Q_VALUES)))
        all_fr_targeted = np.zeros((N_REALIZATIONS, len(Q_VALUES)))
        all_ar = []

        for r in range(N_REALIZATIONS):
            seed = BASE_SEED + r
            G = generate_network(topo, seed)
            rng = np.random.RandomState(seed)

            # Part A: Percolation sweep
            p_inf = bond_percolation_sweep(G, P_RANGE, rng)
            chi = compute_susceptibility(P_RANGE, p_inf)
            pc_sim = find_percolation_threshold(P_RANGE, p_inf)
            pc_theo = theoretical_pc(G)
            p_peak = P_RANGE[np.argmax(chi)]
            chi_max = np.max(chi)

            all_p_inf[r] = p_inf
            all_pc_sim.append(pc_sim)
            all_pc_theo.append(pc_theo)
            all_p_peak.append(p_peak)
            all_chi_max.append(chi_max)

            # Part B: Robustness
            fr_rand = robustness_random_removal(G, Q_VALUES, N_RANDOM_TRIALS, rng)
            fr_targ = robustness_targeted_removal(G, Q_VALUES)

            all_fr_random[r] = fr_rand
            all_fr_targeted[r] = fr_targ

            # A_r at q = 0.28
            q_idx = np.argmin(np.abs(Q_VALUES - Q_REPORT))
            ar = fr_targ[q_idx] / max(fr_rand[q_idx], 1e-10)
            all_ar.append(ar)

            if (r + 1) % 20 == 0:
                print(f"  Realization {r+1}/{N_REALIZATIONS}...")

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Compute transition width Δp (P_∞ from 0.1 to 0.9)
        mean_p_inf = all_p_inf.mean(axis=0)
        p01_idx = np.where(mean_p_inf >= 0.1)[0]
        p09_idx = np.where(mean_p_inf >= 0.9)[0]
        p01 = P_RANGE[p01_idx[0]] if len(p01_idx) > 0 else 0.0
        p09 = P_RANGE[p09_idx[0]] if len(p09_idx) > 0 else 1.0
        delta_p = p09 - p01

        q_idx = np.argmin(np.abs(Q_VALUES - Q_REPORT))

        results[topo] = {
            'pc_sim': (np.mean(all_pc_sim), np.std(all_pc_sim)),
            'pc_theo': (np.mean(all_pc_theo), np.std(all_pc_theo)),
            'p_peak': (np.mean(all_p_peak), np.std(all_p_peak)),
            'chi_max': (np.mean(all_chi_max), np.std(all_chi_max)),
            'delta_p': (delta_p, 0.0),
            'fr_random_028': (all_fr_random[:, q_idx].mean(),
                              all_fr_random[:, q_idx].std()),
            'fr_targeted_028': (all_fr_targeted[:, q_idx].mean(),
                                all_fr_targeted[:, q_idx].std()),
            'A_r': (np.mean(all_ar), np.std(all_ar)),
        }

        # Store curves for figure generation
        ensemble_data[f'{topo}_p_inf_mean'] = mean_p_inf
        ensemble_data[f'{topo}_p_inf_std'] = all_p_inf.std(axis=0)
        ensemble_data[f'{topo}_chi_mean'] = compute_susceptibility(P_RANGE, mean_p_inf)
        ensemble_data[f'{topo}_fr_random_mean'] = all_fr_random.mean(axis=0)
        ensemble_data[f'{topo}_fr_targeted_mean'] = all_fr_targeted.mean(axis=0)
        ensemble_data[f'{topo}_ar_values'] = np.array(all_ar)

    ensemble_data['p_range'] = P_RANGE
    ensemble_data['q_values'] = Q_VALUES
    return results, ensemble_data


def print_results(results):
    print("\n" + "=" * 70)
    print("TABLE S5b — Percolation dynamics and robustness-fragility")
    print(f"(N = {N}, ⟨k⟩ ≈ 6, {N_REALIZATIONS} realizations)")
    print("=" * 70)

    header = f"{'Measure':<22} {'ER':<20} {'WS':<20} {'BA':<20}"
    print(header)
    print("-" * 82)

    measures = [
        ('p_c (P_∞ ≥ 0.5)', 'pc_sim'),
        ('p_c (theoretical)', 'pc_theo'),
        ('p_peak', 'p_peak'),
        ('χ_max', 'chi_max'),
        ('Δp (0.1→0.9)', 'delta_p'),
        ('f_r(random, 0.28)', 'fr_random_028'),
        ('f_r(targeted, 0.28)', 'fr_targeted_028'),
        ('A_r(0.28)', 'A_r'),
    ]

    for label, key in measures:
        row = f"{label:<22}"
        for topo in ['ER', 'WS', 'BA']:
            m, s = results[topo][key]
            if s > 0:
                row += f" {m:>7.3f} ± {s:<6.3f}  "
            else:
                row += f" {m:>7.3f}            "
        print(row)

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION against manuscript Table 2 / Table S5b")
    print("=" * 70)
    expected = {
        'ER': {'p_peak': 0.22, 'A_r': 1.14},
        'WS': {'p_peak': 0.27, 'A_r': 1.02},
        'BA': {'p_peak': 0.16, 'A_r': 2.92},
    }
    for topo in ['ER', 'WS', 'BA']:
        pp_sim = results[topo]['p_peak'][0]
        ar_sim = results[topo]['A_r'][0]
        pp_exp = expected[topo]['p_peak']
        ar_exp = expected[topo]['A_r']
        print(f"  {topo}: p_peak expected={pp_exp:.2f} sim={pp_sim:.2f} | "
              f"A_r expected={ar_exp:.2f} sim={ar_sim:.2f}")


def save_results(results, ensemble_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Percolation CSV
    csv_path = os.path.join(OUTPUT_DIR, 'percolation_results.csv')
    with open(csv_path, 'w') as f:
        f.write("topology,measure,mean,sd\n")
        for topo in ['ER', 'WS', 'BA']:
            for key, (mean, sd) in results[topo].items():
                f.write(f"{topo},{key},{mean:.6f},{sd:.6f}\n")
    print(f"\nSaved: {csv_path}")

    # Ensemble NPZ
    npz_path = os.path.join(OUTPUT_DIR, 'percolation_ensemble_data.npz')
    np.savez_compressed(npz_path, **ensemble_data)
    print(f"Saved: {npz_path}")


if __name__ == '__main__':
    results, ensemble_data = run_ensemble()
    print_results(results)
    save_results(results, ensemble_data)
    print("\n✓ 03_percolation_robustness.py complete.")
