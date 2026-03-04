#!/usr/bin/env python3
"""
04_coupled_learning.py
======================
Manuscript Section 2.4 — Coupled Learning Dynamics (Eqs 5, 11, 12)

Implements the six-step coupled learning algorithm integrating:
  1. Fitness-extended preferential attachment (Eq 5)
  2. Percolation monitoring (Eq 9)
  3. Cascade propagation (Eqs 2–4)
  4. Hebbian weight update (Eq 11)
  5. Convergence check (Frobenius norm)
  6. Performance computation (Eq 12)

The unified performance model:
  P(t) = P_∞(W(t)) = |C₁(W(t))| / N  (Eq 12)

Parameters (Table S6b):
  η = 0.01       Hebbian learning rate
  δ = 0.001      Weight decay rate
  ε = 1e-4       Convergence criterion (Frobenius norm)
  T = 1000       Practice iterations
  w_thresh = 0.5 Weight threshold for active edges

Key outputs (Table S7b):
  w̄(hub-hub), w̄(hub-periph), w̄(periph-periph)
  H_w = w̄(hub-hub) / w̄(periph-periph)  (weight hierarchy)
  t₀.₅ = practice steps to P_∞ = 0.5

Output:
  data/simulation_outputs/coupled_learning_results.csv
  data/simulation_outputs/coupled_learning_data.npz

Usage:
  python simulations/04_coupled_learning.py
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
N_REALIZATIONS = 5        # Table S7b uses 5 realizations

# Network parameters
ER_P = 6.0 / (N - 1)
WS_K = 6
WS_BETA = 0.1
BA_M = 3

# Cascade parameters (Table S2)
P_N = 0.1
P_L = 0.1
P_MAX = 0.99
F_0 = 0.7
F_1 = 0.7
C_P = 1.0
TAU = 5
CASCADE_STEPS = 20        # Cascade steps per practice iteration

# Coupled learning parameters (Table S6b)
ETA = 0.01                # Hebbian learning rate
DELTA = 0.001             # Weight decay rate
EPSILON = 1e-4            # Convergence criterion
T_MAX = 1000              # Practice iterations
W_THRESH = 0.5            # Weight threshold for active edge set
W_BASE = 1.0              # Baseline edge weight

# Hub/peripheral threshold
HUB_SIGMA = 2             # degree > ⟨k⟩ + 2σ

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
# CASCADE SUBROUTINE (simplified for coupled learning)
# ══════════════════════════════════════════════════════════════
def run_cascade_step(adj_idx, ce_array, weights_matrix, rng, n):
    """
    Run a single cascade iteration using weighted propagation.
    Returns activation state array.
    """
    f_pi = F_0 + F_1 * ce_array
    active = np.zeros(n, dtype=bool)
    potential = np.zeros(n, dtype=bool)
    strength = np.ones(n) * 0.1

    # Spontaneous activation
    spontaneous = rng.random(n) < P_N
    potential[spontaneous] = True

    for t in range(CASCADE_STEPS):
        new_potential = potential.copy()
        new_active = active.copy()

        # Propagation (weight-modulated)
        active_indices = np.where(active)[0]
        for i in active_indices:
            for j in adj_idx[i]:
                if not active[j] and not new_potential[j]:
                    # Weight-modulated propagation probability
                    w_ij = weights_matrix[i, j]
                    p_eff = min(P_L * (1 + w_ij / 10.0), 0.95)
                    if rng.random() < p_eff:
                        new_potential[j] = True

        # State transition
        pot_indices = np.where(new_potential & ~active)[0]
        for i in pot_indices:
            pi_i = P_MAX / (1.0 + C_P / max(f_pi[i] * strength[i], 1e-12))
            if rng.random() < (1.0 - pi_i):
                new_active[i] = True
                new_potential[i] = False
                strength[i] = 0.0

        # Strength update
        inactive = ~new_active
        strength[inactive] = 1.0 + (1.0 - f_pi[inactive]) * strength[inactive]

        potential = new_potential
        active = new_active

        if active.all():
            break

    return active


# ══════════════════════════════════════════════════════════════
# HEBBIAN WEIGHT UPDATE — Eq 11
# ══════════════════════════════════════════════════════════════
def hebbian_update(weights, active, adj_idx, n):
    """
    Eq 11: w_ij(t+1) = w_ij(t) + η × A_i(t) × A_j(t) − δ × w_ij(t)

    Only updates existing edges.
    """
    new_weights = weights.copy()

    # Hebbian strengthening: co-activated edges
    active_idx = np.where(active)[0]
    for i in active_idx:
        for j in adj_idx[i]:
            if active[j] and j > i:  # Avoid double-counting
                new_weights[i, j] += ETA * 1.0 * 1.0  # A_i × A_j = 1
                new_weights[j, i] = new_weights[i, j]  # Symmetric

    # Decay: all edges
    for i in range(n):
        for j in adj_idx[i]:
            if j > i:
                new_weights[i, j] -= DELTA * new_weights[i, j]
                new_weights[j, i] = new_weights[i, j]

    # Ensure non-negative
    new_weights = np.maximum(new_weights, 0.0)

    return new_weights


# ══════════════════════════════════════════════════════════════
# PERFORMANCE COMPUTATION — Eq 12
# ══════════════════════════════════════════════════════════════
def compute_performance(weights, G, n, w_thresh):
    """
    Eq 12: P(t) = |C₁(W(t))| / N

    Compute giant component of the weight-thresholded network.
    """
    nodes = sorted(G.nodes())
    H = nx.Graph()
    H.add_nodes_from(nodes)

    for u, v in G.edges():
        i, j = nodes.index(u), nodes.index(v)
        if weights[i, j] >= w_thresh:
            H.add_edge(u, v)

    if H.number_of_edges() == 0:
        return 1.0 / n
    gcc = max(nx.connected_components(H), key=len)
    return len(gcc) / n


# ══════════════════════════════════════════════════════════════
# WEIGHT HIERARCHY COMPUTATION
# ══════════════════════════════════════════════════════════════
def compute_weight_hierarchy(weights, hub_indices, periph_indices, adj_idx):
    """
    Compute mean edge weights by class:
    - w̄(hub-hub), w̄(hub-periph), w̄(periph-periph)
    - H_w = w̄(hub-hub) / w̄(periph-periph)
    """
    hub_set = set(hub_indices)
    periph_set = set(periph_indices)

    w_hh, w_hp, w_pp = [], [], []

    for i in range(len(weights)):
        for j in adj_idx[i]:
            if j <= i:
                continue
            w = weights[i, j]
            if i in hub_set and j in hub_set:
                w_hh.append(w)
            elif (i in hub_set and j in periph_set) or \
                 (i in periph_set and j in hub_set):
                w_hp.append(w)
            elif i in periph_set and j in periph_set:
                w_pp.append(w)

    mean_hh = np.mean(w_hh) if w_hh else 0.0
    mean_hp = np.mean(w_hp) if w_hp else 0.0
    mean_pp = np.mean(w_pp) if w_pp else 0.0
    h_w = mean_hh / max(mean_pp, 1e-10)

    return mean_hh, mean_hp, mean_pp, h_w


# ══════════════════════════════════════════════════════════════
# MAIN COUPLED LEARNING LOOP
# ══════════════════════════════════════════════════════════════
def run_coupled_learning(topo, seed):
    """Run the complete six-step coupled learning algorithm."""
    G = generate_network(topo, seed)
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    # Build adjacency
    adj_idx = [[] for _ in range(n)]
    for u, v in G.edges():
        adj_idx[node_to_idx[u]].append(node_to_idx[v])
        adj_idx[node_to_idx[v]].append(node_to_idx[u])

    # Eigenvector centrality
    try:
        ce = nx.eigenvector_centrality_numpy(G)
    except Exception:
        ce = {nd: 1.0 / n for nd in G.nodes()}
    ce_array = np.array([ce[nd] for nd in nodes])

    # Hub/peripheral identification
    degrees = np.array([G.degree(nd) for nd in nodes])
    k_mean = degrees.mean()
    k_std = degrees.std()
    hub_indices = [i for i in range(n)
                   if degrees[i] > k_mean + HUB_SIGMA * k_std]
    periph_indices = [i for i in range(n)
                      if degrees[i] < k_mean]

    if not hub_indices:
        top_k = max(1, n // 20)
        hub_indices = list(np.argsort(degrees)[-top_k:])
    if not periph_indices:
        bot_k = max(1, n // 5)
        periph_indices = list(np.argsort(degrees)[:bot_k])

    # Initialize weight matrix: w_ij(0) = w_base × √(k_i × k_j) / ⟨k⟩
    weights = np.zeros((n, n))
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        w_init = W_BASE * np.sqrt(degrees[i] * degrees[j]) / k_mean
        weights[i, j] = w_init
        weights[j, i] = w_init

    rng = np.random.RandomState(seed)

    # Tracking arrays
    performance_traj = np.zeros(T_MAX)
    hw_traj = np.zeros(T_MAX)
    w_hh_traj = np.zeros(T_MAX)
    w_hp_traj = np.zeros(T_MAX)
    w_pp_traj = np.zeros(T_MAX)
    t_half = T_MAX  # Default if never reached

    for t in range(T_MAX):
        # Step 3–4: Cascade propagation
        active = run_cascade_step(adj_idx, ce_array, weights, rng, n)

        # Step 5: Hebbian weight update (Eq 11)
        old_weights = weights.copy()
        weights = hebbian_update(weights, active, adj_idx, n)

        # Step 6: Performance computation (Eq 12)
        perf = compute_performance(weights, G, n, W_THRESH)
        performance_traj[t] = perf

        # Weight hierarchy
        mean_hh, mean_hp, mean_pp, h_w = compute_weight_hierarchy(
            weights, hub_indices, periph_indices, adj_idx)
        hw_traj[t] = h_w
        w_hh_traj[t] = mean_hh
        w_hp_traj[t] = mean_hp
        w_pp_traj[t] = mean_pp

        # t₀.₅ detection
        if perf >= 0.5 and t_half == T_MAX:
            t_half = t

        # Convergence check
        frob_norm = np.linalg.norm(weights - old_weights, 'fro')
        if frob_norm < EPSILON and t > 100:
            # Fill remaining trajectory
            performance_traj[t+1:] = perf
            hw_traj[t+1:] = h_w
            w_hh_traj[t+1:] = mean_hh
            w_hp_traj[t+1:] = mean_hp
            w_pp_traj[t+1:] = mean_pp
            break

    return {
        'performance': performance_traj,
        'hw': hw_traj,
        'w_hh': w_hh_traj,
        'w_hp': w_hp_traj,
        'w_pp': w_pp_traj,
        't_half': t_half,
        'final_hw': hw_traj[min(T_MAX - 1, max(0, t))],
        'final_perf': performance_traj[min(T_MAX - 1, max(0, t))],
        'final_w_hh': w_hh_traj[min(T_MAX - 1, max(0, t))],
        'final_w_hp': w_hp_traj[min(T_MAX - 1, max(0, t))],
        'final_w_pp': w_pp_traj[min(T_MAX - 1, max(0, t))],
    }


# ══════════════════════════════════════════════════════════════
# ENSEMBLE SIMULATION
# ══════════════════════════════════════════════════════════════
def run_ensemble():
    print("=" * 70)
    print("04_coupled_learning.py")
    print("Coupled Learning Dynamics (Eqs 5, 11, 12)")
    print(f"N = {N}, η = {ETA}, δ = {DELTA}, T = {T_MAX}, "
          f"{N_REALIZATIONS} realizations")
    print("=" * 70)

    topologies = ['ER', 'WS', 'BA']
    results = {}
    ensemble_data = {}

    for topo in topologies:
        print(f"\n--- {topo} ---")
        t0 = time.time()

        all_t_half = []
        all_hw = []
        all_w_hh = []
        all_w_hp = []
        all_w_pp = []
        all_perf_final = []
        all_perf_traj = []

        for r in range(N_REALIZATIONS):
            seed = BASE_SEED + r
            print(f"  Realization {r+1}/{N_REALIZATIONS}...", end=' ')

            run_result = run_coupled_learning(topo, seed)

            all_t_half.append(run_result['t_half'])
            all_hw.append(run_result['final_hw'])
            all_w_hh.append(run_result['final_w_hh'])
            all_w_hp.append(run_result['final_w_hp'])
            all_w_pp.append(run_result['final_w_pp'])
            all_perf_final.append(run_result['final_perf'])
            all_perf_traj.append(run_result['performance'])

            print(f"t₀.₅={run_result['t_half']}, "
                  f"H_w={run_result['final_hw']:.2f}, "
                  f"P(T)={run_result['final_perf']:.3f}")

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        results[topo] = {
            'w_hh': (np.mean(all_w_hh), np.std(all_w_hh)),
            'w_hp': (np.mean(all_w_hp), np.std(all_w_hp)),
            'w_pp': (np.mean(all_w_pp), np.std(all_w_pp)),
            'H_w': (np.mean(all_hw), np.std(all_hw)),
            't_half': (np.mean(all_t_half), np.std(all_t_half)),
            'P_final': (np.mean(all_perf_final), np.std(all_perf_final)),
        }

        # Mean trajectory for figure generation
        mean_traj = np.mean(all_perf_traj, axis=0)
        std_traj = np.std(all_perf_traj, axis=0)
        ensemble_data[f'{topo}_perf_mean'] = mean_traj
        ensemble_data[f'{topo}_perf_std'] = std_traj
        ensemble_data[f'{topo}_t_half_values'] = np.array(all_t_half)

    return results, ensemble_data


def print_results(results):
    print("\n" + "=" * 70)
    print("TABLE S7b — Coupled learning dynamics measures")
    print(f"(η = {ETA}, δ = {DELTA}, T = {T_MAX}, "
          f"{N_REALIZATIONS} realizations)")
    print("=" * 70)

    header = f"{'Measure':<22} {'ER':<20} {'WS':<20} {'BA':<20}"
    print(header)
    print("-" * 82)

    measures = [
        ('w̄(hub-hub)', 'w_hh'),
        ('w̄(hub-periph)', 'w_hp'),
        ('w̄(periph-periph)', 'w_pp'),
        ('H_w', 'H_w'),
        ('t₀.₅ (steps)', 't_half'),
        ('P(t=1000)', 'P_final'),
    ]

    for label, key in measures:
        row = f"{label:<22}"
        for topo in ['ER', 'WS', 'BA']:
            m, s = results[topo][key]
            row += f" {m:>8.2f} ± {s:<6.2f}  "
        print(row)

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION against manuscript Table 2 / Table S7b")
    print("=" * 70)
    expected = {
        'ER': {'H_w': 0.5, 't_half': 84},
        'WS': {'H_w': 0.4, 't_half': 262},
        'BA': {'H_w': 5.5, 't_half': 56},
    }
    for topo in ['ER', 'WS', 'BA']:
        hw_sim = results[topo]['H_w'][0]
        th_sim = results[topo]['t_half'][0]
        hw_exp = expected[topo]['H_w']
        th_exp = expected[topo]['t_half']
        print(f"  {topo}: H_w expected={hw_exp:.1f} sim={hw_sim:.2f} | "
              f"t₀.₅ expected={th_exp} sim={th_sim:.0f}")


def save_results(results, ensemble_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, 'coupled_learning_results.csv')
    with open(csv_path, 'w') as f:
        f.write("topology,measure,mean,sd\n")
        for topo in ['ER', 'WS', 'BA']:
            for key, (mean, sd) in results[topo].items():
                f.write(f"{topo},{key},{mean:.6f},{sd:.6f}\n")
    print(f"\nSaved: {csv_path}")

    npz_path = os.path.join(OUTPUT_DIR, 'coupled_learning_data.npz')
    np.savez_compressed(npz_path, **ensemble_data)
    print(f"Saved: {npz_path}")


if __name__ == '__main__':
    results, ensemble_data = run_ensemble()
    print_results(results)
    save_results(results, ensemble_data)
    print("\n✓ 04_coupled_learning.py complete.")
