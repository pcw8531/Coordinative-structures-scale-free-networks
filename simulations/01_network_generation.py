#!/usr/bin/env python3
"""
01_network_generation.py
========================
Coordinative Structures as Scale-Free Networks (Park, 2026)
Manuscript Section 2.1 — Network Topology

Generates three canonical network topologies with matched connectivity:
  - Erdős–Rényi (ER) random network
  - Watts–Strogatz (WS) small-world network
  - Barabási–Albert (BA) scale-free network

Computes structural metrics for manuscript Tables S1a–S1d and Table 2:
  - Degree statistics: ⟨k⟩, σ_k, k_max
  - Degree heterogeneity: κ = ⟨k²⟩/⟨k⟩
  - Power-law exponent: γ̂ (MLE) with KS goodness-of-fit
  - Centrality: eigenvector centrality, Gini coefficient, hub fraction
  - Topology: clustering coefficient, average path length

Parameters (from manuscript §2.1):
  N = 100 nodes
  ⟨k⟩ ≈ 6 (matched across topologies)
  ER: p = 0.0606
  WS: K = 6, β = 0.1
  BA: m = 3 (m₀ = 3 initial complete graph)
  Realizations: 100
  Base random seed: 42

Output:
  data/simulation_outputs/network_structural_metrics.csv
  data/simulation_outputs/network_ensemble_data.npz

Usage:
  python simulations/01_network_generation.py
"""

import numpy as np
import networkx as nx
from scipy import stats
import os
import sys
import time

# ══════════════════════════════════════════════════════════════
# PARAMETERS — manuscript §2.1, Tables S1b, S1d
# ══════════════════════════════════════════════════════════════
N = 100                   # Number of nodes (degrees of freedom)
N_REALIZATIONS = 100      # Independent network instances
BASE_SEED = 42            # For reproducibility

# Topology-specific parameters (Table S1b)
ER_P = 6.0 / (N - 1)     # ≈ 0.0606, targeting ⟨k⟩ ≈ 6
WS_K = 6                  # Ring neighbors (each side K/2 = 3)
WS_BETA = 0.1             # Rewiring probability
BA_M = 3                  # Edges per new node (⟨k⟩ → 2m = 6)

# Hub identification threshold
HUB_CRITERION = 'mean_plus_sd'  # C_e > μ + σ

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'data', 'simulation_outputs')


# ══════════════════════════════════════════════════════════════
# NETWORK GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════
def generate_er(n, p, seed):
    """Erdős–Rényi G(N,p) random network (Eq S1 Appendix, Algorithm 1)."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


def generate_ws(n, k, beta, seed):
    """Watts–Strogatz small-world network (Algorithm 2)."""
    return nx.watts_strogatz_graph(n, k, beta, seed=seed)


def generate_ba(n, m, seed):
    """Barabási–Albert scale-free network via preferential attachment (Eq 1)."""
    return nx.barabasi_albert_graph(n, m, seed=seed)


# ══════════════════════════════════════════════════════════════
# STRUCTURAL ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════
def compute_degree_stats(G):
    """Compute degree distribution statistics."""
    degrees = np.array([G.degree(n) for n in G.nodes()])
    k_mean = degrees.mean()
    k_std = degrees.std()
    k_max = degrees.max()
    k_second_moment = (degrees ** 2).mean()
    kappa = k_second_moment / k_mean if k_mean > 0 else 0.0
    return {
        'k_mean': k_mean,
        'k_std': k_std,
        'k_max': k_max,
        'kappa': kappa,
        'degrees': degrees,
    }


def compute_centrality_stats(G):
    """Compute eigenvector centrality and derived measures."""
    try:
        ce = nx.eigenvector_centrality_numpy(G)
    except nx.NetworkXError:
        # Fallback for disconnected graphs
        ce = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    ce_vals = np.array([ce[n] for n in G.nodes()])

    # Gini coefficient of eigenvector centrality
    gini = _gini_coefficient(ce_vals)

    # Hub fraction: nodes with C_e > μ + σ
    ce_mean = ce_vals.mean()
    ce_std = ce_vals.std()
    hub_fraction = np.mean(ce_vals > (ce_mean + ce_std))

    return {
        'gini_ce': gini,
        'hub_fraction': hub_fraction,
        'ce_values': ce_vals,
    }


def _gini_coefficient(values):
    """Compute Gini coefficient for a set of values."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0 or sorted_vals.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n


def compute_topology_stats(G):
    """Compute clustering coefficient and average path length."""
    clustering = nx.average_clustering(G)

    # Average shortest path length (handle disconnected graphs)
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        # Use largest connected component
        gcc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(gcc).copy()
        avg_path = nx.average_shortest_path_length(subG) if len(subG) > 1 else float('inf')

    return {
        'clustering': clustering,
        'avg_path_length': avg_path,
    }


def powerlaw_fit(degrees, k_min=None):
    """
    Maximum Likelihood Estimation of power-law exponent γ̂.

    Uses the discrete MLE formula (Eq S1 in S1 Appendix):
        γ̂ = 1 + n [Σᵢ ln(kᵢ / (k_min − 0.5))]⁻¹

    Following Clauset, Shalizi & Newman (2009) [Ref 58 in manuscript].

    Returns:
        gamma_hat: MLE exponent estimate
        ks_pvalue: KS test p-value against power-law distribution
        k_min_used: k_min threshold used
    """
    degrees = np.array(degrees)
    degrees = degrees[degrees > 0]  # Remove zeros

    if k_min is None:
        # Find optimal k_min by minimizing KS statistic
        unique_k = np.unique(degrees)
        unique_k = unique_k[unique_k >= 2]  # Minimum meaningful k_min
        best_ks = np.inf
        best_kmin = 2
        best_gamma = 3.0

        for km in unique_k:
            if km >= degrees.max():
                break
            tail = degrees[degrees >= km]
            if len(tail) < 10:
                break
            g = 1 + len(tail) * (np.sum(np.log(tail / (km - 0.5)))) ** (-1)
            if g > 1.5 and g < 5.0:  # Reasonable range
                # KS statistic
                cdf_emp = np.sort(tail)
                cdf_emp = np.arange(1, len(cdf_emp) + 1) / len(cdf_emp)
                cdf_theo = 1 - (np.sort(tail) / km) ** (-(g - 1))
                ks = np.max(np.abs(cdf_emp - cdf_theo))
                if ks < best_ks:
                    best_ks = ks
                    best_kmin = km
                    best_gamma = g

        k_min = best_kmin

    # Final fit with chosen k_min
    tail = degrees[degrees >= k_min]
    if len(tail) < 5:
        return np.nan, np.nan, k_min

    gamma_hat = 1 + len(tail) * (np.sum(np.log(tail / (k_min - 0.5)))) ** (-1)

    # KS goodness-of-fit via Monte Carlo (simplified)
    n_mc = 500
    ks_orig = _ks_stat_powerlaw(tail, gamma_hat, k_min)
    n_exceed = 0
    rng = np.random.RandomState(BASE_SEED)
    for _ in range(n_mc):
        # Generate synthetic power-law sample
        synth = _generate_powerlaw_sample(len(tail), gamma_hat, k_min, rng)
        if len(synth) < 5:
            continue
        g_synth = 1 + len(synth) * (np.sum(np.log(synth / (k_min - 0.5)))) ** (-1)
        ks_synth = _ks_stat_powerlaw(synth, g_synth, k_min)
        if ks_synth >= ks_orig:
            n_exceed += 1

    ks_pvalue = n_exceed / n_mc

    return gamma_hat, ks_pvalue, k_min


def _ks_stat_powerlaw(data, gamma, k_min):
    """KS statistic between data and power-law CDF."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf_emp = np.arange(1, n + 1) / n
    cdf_theo = 1 - (sorted_data / k_min) ** (-(gamma - 1))
    return np.max(np.abs(cdf_emp - cdf_theo))


def _generate_powerlaw_sample(n, gamma, k_min, rng):
    """Generate discrete power-law random sample via inverse transform."""
    u = rng.uniform(0, 1, size=n)
    samples = np.floor(k_min * (1 - u) ** (-1.0 / (gamma - 1))).astype(int)
    return samples[samples >= k_min]


# ══════════════════════════════════════════════════════════════
# MAIN ENSEMBLE SIMULATION
# ══════════════════════════════════════════════════════════════
def run_ensemble():
    """
    Generate network ensembles and compute all structural metrics.
    Reproduces Tables S1b, S1d, and Table 2 (structural rows).
    """
    print("=" * 70)
    print("01_network_generation.py")
    print("Coordinative Structures as Scale-Free Networks")
    print(f"N = {N}, ⟨k⟩ ≈ 6, {N_REALIZATIONS} realizations, seed = {BASE_SEED}")
    print("=" * 70)

    topologies = {
        'ER': {'generator': lambda seed: generate_er(N, ER_P, seed),
               'params': f'p = {ER_P:.4f}'},
        'WS': {'generator': lambda seed: generate_ws(N, WS_K, WS_BETA, seed),
               'params': f'K = {WS_K}, β = {WS_BETA}'},
        'BA': {'generator': lambda seed: generate_ba(N, BA_M, seed),
               'params': f'm = {BA_M}'},
    }

    # Storage for ensemble results
    results = {}
    ensemble_data = {}

    for topo_name, topo_info in topologies.items():
        print(f"\n--- {topo_name} ({topo_info['params']}) ---")
        t0 = time.time()

        # Per-realization storage
        k_means = []
        k_stds = []
        k_maxs = []
        kappas = []
        clusterings = []
        path_lengths = []
        gini_ces = []
        hub_fracs = []
        gamma_hats = []
        ks_pvals = []
        all_degrees = []

        for r in range(N_REALIZATIONS):
            seed = BASE_SEED + r
            G = topo_info['generator'](seed)

            # Degree statistics
            dstats = compute_degree_stats(G)
            k_means.append(dstats['k_mean'])
            k_stds.append(dstats['k_std'])
            k_maxs.append(dstats['k_max'])
            kappas.append(dstats['kappa'])
            all_degrees.append(dstats['degrees'])

            # Centrality statistics
            cstats = compute_centrality_stats(G)
            gini_ces.append(cstats['gini_ce'])
            hub_fracs.append(cstats['hub_fraction'])

            # Topology statistics
            tstats = compute_topology_stats(G)
            clusterings.append(tstats['clustering'])
            path_lengths.append(tstats['avg_path_length'])

            # Power-law fit (BA only — ER and WS are not power-law)
            if topo_name == 'BA':
                gamma, ks_p, _ = powerlaw_fit(dstats['degrees'])
                gamma_hats.append(gamma)
                ks_pvals.append(ks_p)

        elapsed = time.time() - t0
        print(f"  Completed {N_REALIZATIONS} realizations in {elapsed:.1f}s")

        # Ensemble statistics
        res = {
            'k_mean': (np.mean(k_means), np.std(k_means)),
            'k_std': (np.mean(k_stds), np.std(k_stds)),
            'k_max': (np.mean(k_maxs), np.std(k_maxs)),
            'kappa': (np.mean(kappas), np.std(kappas)),
            'clustering': (np.mean(clusterings), np.std(clusterings)),
            'avg_path': (np.mean(path_lengths), np.std(path_lengths)),
            'gini_ce': (np.mean(gini_ces), np.std(gini_ces)),
            'hub_fraction': (np.mean(hub_fracs), np.std(hub_fracs)),
        }

        if topo_name == 'BA':
            valid_gamma = [g for g in gamma_hats if not np.isnan(g)]
            valid_ks = [p for p in ks_pvals if not np.isnan(p)]
            res['gamma_hat'] = (np.mean(valid_gamma), np.std(valid_gamma))
            res['ks_pvalue'] = (np.mean(valid_ks), np.std(valid_ks))

        results[topo_name] = res

        # Store ensemble arrays for downstream use
        ensemble_data[f'{topo_name}_kappas'] = np.array(kappas)
        ensemble_data[f'{topo_name}_gini'] = np.array(gini_ces)
        ensemble_data[f'{topo_name}_hub_frac'] = np.array(hub_fracs)
        ensemble_data[f'{topo_name}_k_means'] = np.array(k_means)
        ensemble_data[f'{topo_name}_k_stds'] = np.array(k_stds)
        ensemble_data[f'{topo_name}_k_maxs'] = np.array(k_maxs)
        ensemble_data[f'{topo_name}_clusterings'] = np.array(clusterings)
        ensemble_data[f'{topo_name}_path_lengths'] = np.array(path_lengths)

        # Store exemplar network (seed=42) for visualization
        G_exemplar = topo_info['generator'](BASE_SEED)
        pos_exemplar = nx.spring_layout(G_exemplar, seed=BASE_SEED,
                                        k=1.8 / np.sqrt(N), iterations=80)
        deg_exemplar = np.array([G_exemplar.degree(n) for n in G_exemplar.nodes()])
        try:
            ce_exemplar = np.array(list(
                nx.eigenvector_centrality_numpy(G_exemplar).values()))
        except Exception:
            ce_exemplar = np.zeros(N)

        ce_norm = (ce_exemplar - ce_exemplar.min()) / (
            ce_exemplar.max() - ce_exemplar.min() + 1e-10)

        pos_array = np.array([pos_exemplar[n] for n in G_exemplar.nodes()])
        edges_array = np.array(list(G_exemplar.edges()))

        ensemble_data[f'net_{topo_name}_pos'] = pos_array
        ensemble_data[f'net_{topo_name}_degs'] = deg_exemplar
        ensemble_data[f'net_{topo_name}_ce'] = ce_norm
        ensemble_data[f'net_{topo_name}_edges'] = edges_array

    return results, ensemble_data


def print_results(results):
    """Print results in Table S1d format."""
    print("\n" + "=" * 70)
    print("TABLE S1d — Structural validation measures")
    print(f"(ensemble means ± SD; {N_REALIZATIONS} realizations; "
          f"N = {N}, ⟨k⟩ ≈ 6, seed base = {BASE_SEED})")
    print("=" * 70)

    measures = [
        ('⟨k⟩', 'k_mean'),
        ('σ_k', 'k_std'),
        ('k_max', 'k_max'),
        ('κ = ⟨k²⟩/⟨k⟩', 'kappa'),
        ('Clustering', 'clustering'),
        ('Avg path', 'avg_path'),
        ('Gini(C_e)', 'gini_ce'),
        ('Hub fraction', 'hub_fraction'),
    ]

    header = f"{'Measure':<18} {'ER (Random)':<20} {'WS (Small-World)':<20} {'BA (Scale-Free)':<20}"
    print(header)
    print("-" * 78)

    for label, key in measures:
        row = f"{label:<18}"
        for topo in ['ER', 'WS', 'BA']:
            mean, sd = results[topo][key]
            row += f" {mean:>7.2f} ± {sd:<6.2f}   "
        print(row)

    # BA-specific measures
    if 'gamma_hat' in results['BA']:
        mean, sd = results['BA']['gamma_hat']
        print(f"{'γ̂ (MLE)':<18} {'---':<20} {'---':<20} {mean:>7.2f} ± {sd:<6.2f}")
    if 'ks_pvalue' in results['BA']:
        mean, sd = results['BA']['ks_pvalue']
        print(f"{'KS p-value':<18} {'---':<20} {'---':<20} {mean:>7.2f} ± {sd:<6.2f}")

    # Manuscript Table 2 comparison
    print("\n" + "=" * 70)
    print("VERIFICATION against manuscript Table 2")
    print("=" * 70)
    expected = {
        'ER': {'kappa': 6.84, 'gini_ce': 0.26, 'hub_fraction': 0.16},
        'WS': {'kappa': 6.09, 'gini_ce': 0.16, 'hub_fraction': 0.16},
        'BA': {'kappa': 9.67, 'gini_ce': 0.35, 'hub_fraction': 0.10},
    }
    for topo in ['ER', 'WS', 'BA']:
        print(f"\n  {topo}:")
        for key, exp_val in expected[topo].items():
            sim_val = results[topo][key][0]
            match = "✓" if abs(sim_val - exp_val) / (exp_val + 1e-10) < 0.15 else "✗"
            print(f"    {key:<15} expected={exp_val:.2f}  simulated={sim_val:.2f}  {match}")


def save_results(results, ensemble_data):
    """Save results to CSV and NPZ files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV summary
    csv_path = os.path.join(OUTPUT_DIR, 'network_structural_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("topology,measure,mean,sd\n")
        for topo in ['ER', 'WS', 'BA']:
            for key, (mean, sd) in results[topo].items():
                f.write(f"{topo},{key},{mean:.6f},{sd:.6f}\n")
    print(f"\nSaved: {csv_path}")

    # NPZ ensemble data (for downstream scripts and figure generation)
    npz_path = os.path.join(OUTPUT_DIR, 'network_ensemble_data.npz')
    np.savez_compressed(npz_path, **ensemble_data)
    print(f"Saved: {npz_path}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    results, ensemble_data = run_ensemble()
    print_results(results)
    save_results(results, ensemble_data)
    print("\n✓ 01_network_generation.py complete.")
