# Coordinative Structures as Scale-Free Networks

**Simulation code and data for:**

> Coordinative structures as scale-free networks: Cascade and percolation dynamics in motor learning with empirical validation

Submitted to *PLOS Computational Biology*

**Author:** Chulwook Park

**Affiliations:**
- Department of Physical Education, Seoul National University, Seoul, Korea
- Systemic Risk and Resilience, International Institute for Applied Systems Analysis (IIASA), Laxenburg, Austria
- Complexity Science and Evolution, Okinawa Institute of Science and Technology (OIST), Okinawa, Japan

**Contact:** pcw8531@snu.ac.kr

## Overview

This repository contains the complete simulation code and empirical reference data for the manuscript. The study proposes scale-free network topology (Barabási–Albert model) as a mechanistic basis for coordinative structures in motor control, integrating cascade propagation, percolation theory, and Hebbian learning dynamics.

Three canonical network topologies are compared:
- **Erdős–Rényi (ER):** Random network (null model)
- **Watts–Strogatz (WS):** Small-world network
- **Barabási–Albert (BA):** Scale-free network

## Repository Structure

```
Coordinative-structures-scale-free-networks/
├── README.md
├── LICENSE                              # MIT License
├── .gitignore                           # Python gitignore
├── requirements.txt                     # Python dependencies
├── simulations/
│   ├── 01_network_generation.py         # ER, WS, BA network construction (Eq 1)
│   ├── 02_cascade_dynamics.py           # Cascade propagation model (Eqs 2–4)
│   ├── 03_percolation_robustness.py     # Bond percolation and node removal (Eqs 6–7, 9–10)
│   └── 04_coupled_learning.py           # Hebbian weight updates and performance (Eqs 5, 11–12)
└── data/
    └── empirical_reference/             # Empirical validation data from published sources
        ├── README.md                    # Data provenance documentation
        ├── vereijken_1992_coupling.csv  # Inter-joint coupling matrix (Fig 4A)
        ├── bassett_2011_nodes.csv       # Brain network nodes (Fig 4A)
        ├── bassett_2011_edges.csv       # Brain network edges (Fig 4A)
        ├── scholz_schoner_1999_nodes.csv # UCM analysis nodes (Fig 4A)
        ├── scholz_schoner_1999_edges.csv # UCM analysis edges (Fig 4A)
        ├── kelso_1986_phase_transition.csv # Bimanual coordination data (Fig 5E)
        └── liu_2006_learning_parameters.json # Motor learning curve parameters (Fig 5D)
```

## Simulation Parameters

All simulations use the following base parameters (see manuscript Tables 1–3 and Supporting Information Tables S1a–S7b for complete specifications):

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 100 | Number of nodes (degrees of freedom) |
| ⟨k⟩ | ≈ 6 | Mean degree across all topologies |
| Realizations | 100 | Independent network instances per topology |
| Random seed | 42 | Base seed for reproducibility |
| T | 1,000 | Practice iterations (coupled learning) |
| η | 0.01 | Hebbian learning rate |
| δ | 0.001 | Weight decay rate |
| p sweep | [0, 1], step 0.01 | Bond occupation probability range |

### Network-Specific Parameters

| Topology | Parameters |
|----------|------------|
| ER (Random) | p = ⟨k⟩/(N−1) ≈ 0.0606 |
| WS (Small-World) | K = 6 (ring neighbors), β = 0.1 (rewiring probability) |
| BA (Scale-Free) | m = 3 (edges per new node), m₀ = 3 (initial complete graph) |

### Cascade Dynamics Parameters (Table S2)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Spontaneous activation | pₙ | 0.1 |
| Link propagation | p_l | 0.1 |
| Max stability | p_max | 0.99 |
| Baseline stability | f₀ | 0.7 |
| Centrality weight | f₁ | 0.7 |
| Early-stage window | τ | 5 |

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- NetworkX ≥ 2.6
- Matplotlib ≥ 3.5
- SciPy ≥ 1.7

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Reproduction

### Quick Start

```bash
# Clone the repository
git clone https://github.com/pcw8531/Coordinative-structures-scale-free-networks.git
cd Coordinative-structures-scale-free-networks

# Install dependencies
pip install -r requirements.txt

# Run simulations sequentially (each generates CSV and NPZ outputs in data/simulation_outputs/)
python simulations/01_network_generation.py
python simulations/02_cascade_dynamics.py
python simulations/03_percolation_robustness.py
python simulations/04_coupled_learning.py
```

### Script Descriptions

| Script | Manuscript Section | Equations | Tables Reproduced |
|--------|-------------------|-----------|-------------------|
| `01_network_generation.py` | §2.1 Network Topology | Eq 1 | S1a–S1d, Table 2 (structural rows) |
| `02_cascade_dynamics.py` | §2.2 Cascade Dynamics | Eqs 2–4, 8 | S2, S3a, S3b, Table 2 (cascade rows) |
| `03_percolation_robustness.py` | §2.3 Correspondences | Eqs 6, 7, 9, 10 | S4, S5, S5b, Table 2 (percolation rows) |
| `04_coupled_learning.py` | §2.4 Coupled Learning | Eqs 5, 11, 12 | S6b, S7b, Table 2 (learning rows) |

Each script prints verification output comparing simulated values against the published manuscript tables. Outputs are saved to `data/simulation_outputs/` (created at runtime).

### Simulation vs. Visualization Parameters

The manuscript uses two parameter sets:

- **Authoritative (for tables/statistics):** 100 realizations, p step = 0.01, T = 1,000
- **Visual (for figure clarity):** 30 realizations, p step = 0.02, T = 1,000

Both are documented in figure captions and Supporting Information.

## Empirical Reference Data

The `data/empirical_reference/` directory contains network-reinterpreted representations of empirical findings from five published studies, used for model validation in Figures 4–5:

| File | Source | Manuscript Figure |
|------|--------|-------------------|
| `vereijken_1992_coupling.csv` | Vereijken et al. (1992) *Hum Mov Sci* | Fig 4A (ER-like topology) |
| `bassett_2011_*.csv` | Bassett et al. (2011) *PNAS* | Fig 4A (WS-like topology) |
| `scholz_schoner_1999_*.csv` | Scholz & Schöner (1999) *Exp Brain Res* | Fig 4A (BA-like topology) |
| `kelso_1986_phase_transition.csv` | Kelso, Scholz & Schöner (1986) *Phys Lett A* | Fig 5E |
| `liu_2006_learning_parameters.json` | Liu, Mayer-Kress & Newell (2006) *Nonlinear Dynamics Psychol Life Sci* | Fig 5D |

All values are directly traceable to the figure generation code and verified against the original publications. See `data/empirical_reference/README.md` for complete provenance documentation.

## Key Results Summary

The simulations demonstrate that scale-free (BA) networks uniquely reproduce properties of coordinative structures (manuscript Table 2):

| Property | ER | WS | BA |
|----------|----|----|-----|
| Degree heterogeneity (κ) | 6.84 ± 0.38 | 6.09 ± 0.02 | **9.67 ± 0.59** |
| Cascade asymmetry (Δ_hp) | 2.25 | 1.46 | **3.60 ± 0.62** |
| Peak susceptibility (p_peak) | 0.22 | 0.27 | **0.16** |
| Robustness asymmetry (A_r) | 1.14 ± 0.09 | 1.02 ± 0.05 | **2.92 ± 0.31** |
| Weight hierarchy (H_w) | 0.5 | 0.4 | **5.5** |
| Learning half-time (t₀.₅) | 84 | 262 | **56** |

## Correspondence to Manuscript Equations

| Equation | Description | Script |
|----------|-------------|--------|
| Eq 1 | Preferential attachment: Π(i) = kᵢ / Σⱼ kⱼ | `01_network_generation.py` |
| Eqs 2–4 | Cascade dynamics: stability, activation, centrality | `02_cascade_dynamics.py` |
| Eq 5 | Fitness-extended PA: Πᵢ = (kᵢᵅ × ηᵢᵝ) / Σⱼ(kⱼᵅ × ηⱼᵝ) | `04_coupled_learning.py` |
| Eq 6 | Percolation threshold: p_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩) | `03_percolation_robustness.py` |
| Eq 7 | Fragmentation ratio: f_r(q) = 1 − S(q)/S(0) | `03_percolation_robustness.py` |
| Eqs 9–10 | Order parameter P_∞ and susceptibility χ | `03_percolation_robustness.py` |
| Eq 11 | Hebbian update: w_ij(t+1) = w_ij(t) + η·Aᵢ·Aⱼ − δ·w_ij(t) | `04_coupled_learning.py` |
| Eq 12 | Unified performance: P(t) = \|C₁(W(t))\| / N | `04_coupled_learning.py` |

## Citation

If you use this code or data, please cite:

```
Park C. Coordinative Structures as Scale-Free Networks: How Cascade
and Percolation Dynamics Shape Motor Learning. PLOS Computational Biology.
2026. [DOI to be added upon publication]
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or manuscript, please open an issue or contact Chulwook Park (pcw8531@snu.ac.kr).
