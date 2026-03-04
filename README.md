# Coordinative Structures as Scale-Free Networks

**Simulation code and data for:**

> Coordinative Structures as Scale-Free Networks: How Cascade and Percolation Dynamics Shape Motor Learning

Submitted to *PLOS Computational Biology*

**Author:** Chulwook Park

**Affiliations:**
- Department of Physical Education, Seoul National University, Seoul, Korea
- Systemic Risk and Resilience, International Institute for Applied Systems Analysis (IIASA), Laxenburg, Austria
- Complexity Science and Evolution, Okinawa Institute of Science and Technology (OIST), Okinawa, Japan

**Contact:** pcw8531@snu.ac.kr

## Overview

This repository contains the complete simulation code, data generation scripts, and figure reproduction materials for the manuscript. The study proposes scale-free network topology (Barabási–Albert model) as a mechanistic basis for coordinative structures in motor control, integrating cascade propagation, percolation theory, and Hebbian learning dynamics.

Three canonical network topologies are compared:
- **Erdős–Rényi (ER):** Random network (null model)
- **Watts–Strogatz (WS):** Small-world network
- **Barabási–Albert (BA):** Scale-free network

## Repository Structure

```
coordinative-structures-scale-free-networks/
├── README.md
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── simulations/
│   ├── 01_network_generation.py     # ER, WS, BA network construction
│   ├── 02_cascade_dynamics.py       # Cascade propagation model (Eqs 2–4)
│   ├── 03_percolation_robustness.py # Bond percolation and node removal (Eqs 6–7, 9–10)
│   └── 04_coupled_learning.py       # Hebbian weight updates and performance (Eqs 5, 11–12)
├── figures/
│   └── generate_figures.py          # Reproduces all manuscript figures (Figs 1–7)
├── data/
│   └── simulation_outputs/          # Pre-computed CSV outputs from 100-realization runs
└── notebooks/
    └── full_analysis.ipynb          # Complete analysis notebook
```

## Simulation Parameters

All simulations use the following base parameters (see manuscript Tables 1–3 and Supporting Information Tables S1a–S8 for complete specifications):

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
| w_thresh | 0.5 | Weight threshold for active edges |

### Network-Specific Parameters

| Topology | Parameters |
|----------|------------|
| ER (Random) | p = ⟨k⟩/(N−1) ≈ 0.0606 |
| WS (Small-World) | K = 6 (ring neighbors), β = 0.1 (rewiring probability) |
| BA (Scale-Free) | m = 3 (edges per new node), m₀ = 3 (initial complete graph) |

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- NetworkX ≥ 2.6
- Matplotlib ≥ 3.5
- SciPy ≥ 1.7
- python-ternary ≥ 1.0.8

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

# Run all simulations (generates data/simulation_outputs/)
python simulations/01_network_generation.py
python simulations/02_cascade_dynamics.py
python simulations/03_percolation_robustness.py
python simulations/04_coupled_learning.py

# Generate all figures
python figures/generate_figures.py
```

### Script Descriptions

| Script | Manuscript Section | Output | Figures/Tables |
|--------|-------------------|--------|----------------|
| `01_network_generation.py` | §2.1 Network Topology | Network objects, degree distributions, structural metrics | Fig 1, Fig 2, Tables S1a–S1d |
| `02_cascade_dynamics.py` | §2.2 Cascade Dynamics | Cascade trajectories, R_early, R_∞, Δ_hp | Fig 3, Tables S2–S3 |
| `03_percolation_robustness.py` | §2.3 Percolation Model | p_c, χ(p), fragmentation ratios, robustness asymmetry | Fig 4, Tables S4–S5 |
| `04_coupled_learning.py` | §2.4 Coupled Learning | P_∞(t) trajectories, weight matrices, hub weight ratio | Fig 5, Fig 6, Tables S6–S8 |
| `generate_figures.py` | All | Publication-quality figures (Figs 1–7) | — |

### Simulation vs. Visualization Parameters

The manuscript uses two parameter sets:

- **Authoritative (for tables/statistics):** 100 realizations, p step = 0.01, T = 1,000
- **Visual (for figure clarity):** 30 realizations, p step = 0.02, T = 1,000

Both are documented in figure captions and Supporting Information.

## Key Results Summary

The simulations demonstrate that scale-free (BA) networks uniquely reproduce properties of coordinative structures:

| Property | ER | WS | BA |
|----------|----|----|-----|
| Degree heterogeneity (κ) | 6.84 | 6.09 | **9.67** |
| Hub cascade reach (R_early) | 4.32 | 4.98 | **7.41** |
| Cascade asymmetry (Δ_hp) | 1.24 | 1.47 | **3.60** |
| Percolation threshold (p_c) | 0.221 | 0.244 | **0.208** |
| Robustness asymmetry (A_r) | 1.08 | 1.12 | **3.42** |
| Learning half-time (t₀.₅) | 84 | 152 | **56** |
| Hub weight ratio (H_w) | 1.8 | 2.1 | **5.5** |

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
