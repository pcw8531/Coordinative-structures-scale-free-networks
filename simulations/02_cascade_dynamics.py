#!/usr/bin/env python3
"""
02_cascade_dynamics.py
======================
Manuscript Section 2.2 — Cascade Dynamics (Eqs 2–4)

Simulates DOF recruitment cascades on three network topologies,
implementing the six-step update cycle from S2 Appendix:
  1. Spontaneous potential generation (probability p_n per node)
  2. Neighbor-mediated propagation (probability p_l per active link)
  3. Stability evaluation (Eq 2)
  4. State transition (Eq 3)
  5. Coordination strength update
  6. Cascade termination check

Computes cascade measures for Tables S3a, S3b, and Table 2:
  - R_early = |A(τ=5)| / |A(0)|  (early propagation rate)
  - R_∞ = |A(t→∞)| / N  (cascade reach)
  - Δ_hp = R_early(hub) / R_early(peripheral)  (asymmetry ratio)

Parameters (Table S2):
  p_n = 0.1       Spontaneous activation probability
  p_l = 0.1       Link propagation probability
  p_max = 0.99    Maximum stability
  f_0 = 0.7       Baseline stability
  f_1 = 0.7       Centrality weight
  c_p = 1.0       Reference constant
  τ = 5           Early-stage window

For topology-isolated trials (Table S3b): p_n = 0, single seed.

Output:
  data/simulation_outputs/cascade_dynamics_results.csv
  data/simulation_outputs/cascade_ensemble_data.npz

Usage:
  python simulations/02_cascade_dynamics.py
"""

import numpy as np
import networkx as nx
import os
import sys
import time

# ══════════════════════════════════════════════════════════════
# PARAMETERS — Table S2 (S2 Appendix)
# ══════════════════════════════════════════════════════════════
# Network parameters (from 01_network_generation.py)
N = 100
BASE_SEED = 42
N_REALIZATIONS = 100
N_CASCADE_TRIALS = 100    # Topology-isolated trials per realization

# Cascade model parameters (Table S2)
P_N = 0.1          # Spontaneous activation probability
P_L = 0.1          # Link propagation probability
P_MAX = 0.99       # Maximum stability (protection probability)
F_0 = 0.7          # Baseline stability floor
F_1 = 0.7          # Eigenvector centrality contribution
C_P = 1.0          # Stability function scaling constant
TAU = 5            # Early-stage window for R_early
T_MAX = 50         # Maximum cascade timesteps
F_M = 0.0          # Memory decay factor (implicit in strength reset)

# Network generation parameters
ER_P = 6.0 / (N - 1)
WS_K = 6
WS_BETA = 0.1
BA_M = 3

# Hub/peripheral definitions
HUB_THRESHOLD_SIGMA = 2   # Hub: degree > ⟨k⟩ + 2σ
# Peripheral: degree < ⟨k⟩

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
# CASCADE DYNAMICS MODEL — Eqs 2–4
# ══════════════════════════════════════════════════════════════
def stability_function(f_pi, s):
    """
    Eq 2: π_i = p_max / [1 + c_p / (f_π_i × s_i)]

    Returns protection probability for a DOF.
    When f_pi * s → 0:  π → 0 (no stability)
    When f_pi * s → ∞:  π → p_max
    At f_pi * s = c_p:  π = p_max / 2
    """
    denom = f_pi * s
    if denom < 1e-12:
        return 0.0
    return P_MAX / (1.0 + C_P / denom)


def compute_f_pi(centrality_i):
    """Eq 4: f_π_i = f_0 + f_1 × C_i"""
    return F_0 + F_1 * centrality_i


def run_single_cascade(G, ce_dict, seed_node, rng, use_spontaneous=False):
    """
    Run a single cascade from a seed node through the six-step update.

    For topology-isolated trials (Table S3b): use_spontaneous=False, p_n=0.

    Returns:
        trajectory: list of |A(t)| at each timestep
    """
    n_nodes = G.number_of_nodes()
    nodes = list(G.nodes())

    # State variables per node
    potential = np.zeros(n_nodes, dtype=int)     # φ_i(t) ∈ {0,1}
    active = np.zeros(n_nodes, dtype=int)        # A_i(t) ∈ {0,1}
    strength = np.ones(n_nodes) * 0.1            # s_i(t) ≥ 0

    # Precompute centrality-based stability fraction (Eq 4)
    f_pi = np.array([compute_f_pi(ce_dict.get(n, 0.0)) for n in nodes])

    # Adjacency list for efficiency
    adj = {n: list(G.neighbors(n)) for n in nodes}

    # Initialize: seed node has potential
    seed_idx = nodes.index(seed_node)
    potential[seed_idx] = 1

    trajectory = [1]  # |A(0)| = 1 (seed has potential)

    for t in range(1, T_MAX + 1):
        new_potential = potential.copy()
        new_active = active.copy()

        # Step 1: Spontaneous activation (probability p_n per node)
        if use_spontaneous:
            for i in range(n_nodes):
                if not active[i] and not potential[i]:
                    if rng.random() < P_N:
                        new_potential[i] = 1

        # Step 2: Neighbor-mediated propagation (p_l per link)
        for i in range(n_nodes):
            if active[i]:
                for j_node in adj[nodes[i]]:
                    j = nodes.index(j_node) if j_node not in nodes[:n_nodes] else j_node
                    # Use node index directly
                    j_idx = list(G.nodes()).index(j_node)
                    if not active[j_idx] and not new_potential[j_idx]:
                        if rng.random() < P_L:
                            new_potential[j_idx] = 1

        # Step 3–4: Stability evaluation (Eq 2) and state transition (Eq 3)
        for i in range(n_nodes):
            if new_potential[i] and not active[i]:
                pi_i = stability_function(f_pi[i], strength[i])
                # Eq 3: P(A_i = 1 | φ_i > 0) = 1 − π_i
                if rng.random() < (1.0 - pi_i):
                    new_active[i] = 1
                    new_potential[i] = 0
                    strength[i] = 0.0  # Reset upon activation

        # Step 5: Coordination strength update (for non-active nodes)
        for i in range(n_nodes):
            if not new_active[i]:
                strength[i] = 1.0 + (1.0 - F_M - f_pi[i]) * strength[i]

        potential = new_potential
        active = new_active
        trajectory.append(int(active.sum()))

        # Step 6: Cascade termination check
        if active.sum() == n_nodes:
            # Pad remaining timesteps
            for _ in range(t + 1, T_MAX + 1):
                trajectory.append(n_nodes)
            break
        if potential.sum() == 0 and (active.sum() == trajectory[-2] if len(trajectory) > 1 else True):
            # No more potential to spread — check if cascade stalled
            any_can_spread = False
            for i in range(n_nodes):
                if active[i]:
                    for j_node in adj[nodes[i]]:
                        j_idx = list(G.nodes()).index(j_node)
                        if not active[j_idx]:
                            any_can_spread = True
                            break
                if any_can_spread:
                    break
            if not any_can_spread and potential.sum() == 0:
                for _ in range(t + 1, T_MAX + 1):
                    trajectory.append(int(active.sum()))
                break

    return trajectory


def run_single_cascade_fast(G, ce_array, seed_idx, rng):
    """
    Optimized cascade for topology-isolated trials (p_n = 0).
    Uses numpy indexing for speed.
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build adjacency as index lists
    adj_idx = [[] for _ in range(n)]
    for u, v in G.edges():
        adj_idx[node_to_idx[u]].append(node_to_idx[v])
        adj_idx[node_to_idx[v]].append(node_to_idx[u])

    # Stability fractions
    f_pi = F_0 + F_1 * ce_array

    # State
    active = np.zeros(n, dtype=bool)
    potential = np.zeros(n, dtype=bool)
    strength = np.ones(n) * 0.1

    potential[seed_idx] = True
    trajectory = [1]

    for t in range(1, T_MAX + 1):
        new_potential = potential.copy()
        new_active = active.copy()

        # Step 2: Propagation from active nodes (no spontaneous: p_n = 0)
        active_indices = np.where(active)[0]
        for i in active_indices:
            for j in adj_idx[i]:
                if not active[j] and not new_potential[j]:
                    if rng.random() < P_L:
                        new_potential[j] = True

        # Steps 3–4: State transition
        pot_indices = np.where(new_potential & ~active)[0]
        for i in pot_indices:
            pi_i = stability_function(f_pi[i], strength[i])
            if rng.random() < (1.0 - pi_i):
                new_active[i] = True
                new_potential[i] = False
                strength[i] = 0.0

        # Step 5: Strength accumulation
        inactive = ~new_active
        strength[inactive] = 1.0 + (1.0 - f_pi[inactive]) * strength[inactive]

        potential = new_potential
        active = new_active
        n_active = int(active.sum())
        trajectory.append(n_active)

        # Step 6: Termination
        if n_active == n:
            trajectory.extend([n] * (T_MAX - t))
            break

    return trajectory


# ══════════════════════════════════════════════════════════════
# CASCADE MEASURES
# ══════════════════════════════════════════════════════════════
def compute_cascade_measures(trajectory, n_nodes):
    """Compute R_early and R_∞ from a trajectory."""
    # R_early = |A(τ)| / |A(0)|, with |A(0)| = 1
    r_early = trajectory[min(TAU, len(trajectory) - 1)]
    r_inf = trajectory[-1] / n_nodes
    return r_early, r_inf


def identify_hub_peripheral(G):
    """
    Identify hub and peripheral nodes.
    Hub: degree > ⟨k⟩ + 2σ
    Peripheral: degree < ⟨k⟩
    """
    degrees = np.array([G.degree(n) for n in sorted(G.nodes())])
    k_mean = degrees.mean()
    k_std = degrees.std()
    nodes = sorted(G.nodes())

    hubs = [nodes[i] for i in range(len(nodes))
            if degrees[i] > k_mean + HUB_THRESHOLD_SIGMA * k_std]
    peripherals = [nodes[i] for i in range(len(nodes))
                   if degrees[i] < k_mean]
    return hubs, peripherals


# ══════════════════════════════════════════════════════════════
# MAIN ENSEMBLE SIMULATION
# ══════════════════════════════════════════════════════════════
def run_ensemble():
    """
    Run topology-isolated cascade trials across all topologies.
    Reproduces Tables S3b and Table 2 (cascade rows).
    """
    print("=" * 70)
    print("02_cascade_dynamics.py")
    print("Cascade Dynamics Model (Eqs 2–4)")
    print(f"N = {N}, τ = {TAU}, p_l = {P_L}, {N_REALIZATIONS} realizations "
          f"× {N_CASCADE_TRIALS} trials")
    print("=" * 70)

    topologies = ['ER', 'WS', 'BA']
    results = {}
    ensemble_data = {}

    for topo in topologies:
        print(f"\n--- {topo} ---")
        t0 = time.time()

        all_r_early_hub = []
        all_r_early_periph = []
        all_r_inf_hub = []
        all_r_inf_periph = []
        all_delta_hp = []

        for r in range(N_REALIZATIONS):
            seed = BASE_SEED + r
            G = generate_network(topo, seed)
            nodes = sorted(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}

            # Eigenvector centrality
            try:
                ce = nx.eigenvector_centrality_numpy(G)
            except Exception:
                ce = {n: 1.0 / N for n in G.nodes()}
            ce_array = np.array([ce[n] for n in nodes])

            hubs, peripherals = identify_hub_peripheral(G)

            if not hubs:
                # Fallback: use top 5% by degree
                degrees = np.array([G.degree(n) for n in nodes])
                top_k = max(1, N // 20)
                hub_indices = np.argsort(degrees)[-top_k:]
                hubs = [nodes[i] for i in hub_indices]

            if not peripherals:
                degrees = np.array([G.degree(n) for n in nodes])
                bot_k = max(1, N // 5)
                periph_indices = np.argsort(degrees)[:bot_k]
                peripherals = [nodes[i] for i in periph_indices]

            rng = np.random.RandomState(seed)

            # Hub-initiated cascades
            r_early_hub_trials = []
            r_inf_hub_trials = []
            for trial in range(min(N_CASCADE_TRIALS, len(hubs) * 20)):
                hub_node = hubs[trial % len(hubs)]
                hub_idx = node_to_idx[hub_node]
                traj = run_single_cascade_fast(G, ce_array, hub_idx, rng)
                re, ri = compute_cascade_measures(traj, N)
                r_early_hub_trials.append(re)
                r_inf_hub_trials.append(ri)

            # Peripheral-initiated cascades
            r_early_periph_trials = []
            r_inf_periph_trials = []
            for trial in range(min(N_CASCADE_TRIALS, len(peripherals) * 5)):
                periph_node = peripherals[trial % len(peripherals)]
                periph_idx = node_to_idx[periph_node]
                traj = run_single_cascade_fast(G, ce_array, periph_idx, rng)
                re, ri = compute_cascade_measures(traj, N)
                r_early_periph_trials.append(re)
                r_inf_periph_trials.append(ri)

            mean_re_hub = np.mean(r_early_hub_trials)
            mean_re_periph = np.mean(r_early_periph_trials)
            mean_ri_hub = np.mean(r_inf_hub_trials)
            mean_ri_periph = np.mean(r_inf_periph_trials)

            delta_hp = mean_re_hub / max(mean_re_periph, 1e-10)

            all_r_early_hub.append(mean_re_hub)
            all_r_early_periph.append(mean_re_periph)
            all_r_inf_hub.append(mean_ri_hub)
            all_r_inf_periph.append(mean_ri_periph)
            all_delta_hp.append(delta_hp)

            if (r + 1) % 20 == 0:
                print(f"  Realization {r+1}/{N_REALIZATIONS}...")

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        results[topo] = {
            'R_early_hub': (np.mean(all_r_early_hub), np.std(all_r_early_hub)),
            'R_early_periph': (np.mean(all_r_early_periph), np.std(all_r_early_periph)),
            'R_inf_hub': (np.mean(all_r_inf_hub), np.std(all_r_inf_hub)),
            'R_inf_periph': (np.mean(all_r_inf_periph), np.std(all_r_inf_periph)),
            'Delta_hp': (np.mean(all_delta_hp), np.std(all_delta_hp)),
        }

        ensemble_data[f'{topo}_delta_hp'] = np.array(all_delta_hp)
        ensemble_data[f'{topo}_r_early_hub'] = np.array(all_r_early_hub)
        ensemble_data[f'{topo}_r_early_periph'] = np.array(all_r_early_periph)

    return results, ensemble_data


def print_results(results):
    """Print results matching Table S3b format."""
    print("\n" + "=" * 70)
    print("TABLE S3b — Cascade dynamics ensemble statistics")
    print(f"({N_REALIZATIONS} realizations × {N_CASCADE_TRIALS} trials; p_n = 0)")
    print("=" * 70)

    header = f"{'Measure':<20} {'ER':<20} {'WS':<20} {'BA':<20}"
    print(header)
    print("-" * 80)

    measures = [
        ('R_early (hub)', 'R_early_hub'),
        ('R_early (periph)', 'R_early_periph'),
        ('R_∞ (hub)', 'R_inf_hub'),
        ('R_∞ (periph)', 'R_inf_periph'),
        ('Δ_hp', 'Delta_hp'),
    ]

    for label, key in measures:
        row = f"{label:<20}"
        for topo in ['ER', 'WS', 'BA']:
            m, s = results[topo][key]
            row += f" {m:>7.2f} ± {s:<6.2f}   "
        print(row)

    # Verification against manuscript Table 2
    print("\n" + "=" * 70)
    print("VERIFICATION against manuscript Table 2 / Table S3b")
    print("=" * 70)
    expected_delta = {'ER': 2.25, 'WS': 1.46, 'BA': 3.60}
    expected_re_hub = {'ER': 7.58, 'WS': 4.98, 'BA': 14.15}
    for topo in ['ER', 'WS', 'BA']:
        dhp_sim = results[topo]['Delta_hp'][0]
        dhp_exp = expected_delta[topo]
        re_sim = results[topo]['R_early_hub'][0]
        re_exp = expected_re_hub[topo]
        print(f"  {topo}: Δ_hp expected={dhp_exp:.2f} simulated={dhp_sim:.2f} | "
              f"R_early(hub) expected={re_exp:.2f} simulated={re_sim:.2f}")


def save_results(results, ensemble_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'cascade_dynamics_results.csv')
    with open(csv_path, 'w') as f:
        f.write("topology,measure,mean,sd\n")
        for topo in ['ER', 'WS', 'BA']:
            for key, (mean, sd) in results[topo].items():
                f.write(f"{topo},{key},{mean:.6f},{sd:.6f}\n")
    print(f"\nSaved: {csv_path}")

    npz_path = os.path.join(OUTPUT_DIR, 'cascade_ensemble_data.npz')
    np.savez_compressed(npz_path, **ensemble_data)
    print(f"Saved: {npz_path}")


if __name__ == '__main__':
    results, ensemble_data = run_ensemble()
    print_results(results)
    save_results(results, ensemble_data)
    print("\n✓ 02_cascade_dynamics.py complete.")
