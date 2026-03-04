# Empirical Reference Data

This directory contains structured data extracted from published empirical studies
used for model validation in the manuscript. Each file documents the source publication,
the specific data represented, and its role in the manuscript's empirical validation.

**Important:** These data represent network-reinterpreted representations of published
findings, constructed to illustrate correspondence between empirical coordination
phenomena and canonical network topologies. The coupling values, edge weights, and
network structures are derived from the qualitative and quantitative patterns reported
in each source publication, not from raw experimental recordings.

## File Inventory

### 1. `vereijken_1992_coupling.csv`
- **Source:** Vereijken B, van Emmerik REA, Whiting HTA, Newell KM. Free(z)ing degrees of freedom in skill acquisition. *J Mot Behav*. 1992;24(1):133–142.
- **Content:** 7×7 symmetric cross-correlation matrix of bilateral body DOFs (L-Ankle, L-Knee, L-Hip, Trunk, R-Hip, R-Knee, R-Ankle) during novice ski-simulator performance.
- **Network interpretation:** Near-uniform coupling (range 0.62–0.82) with all pairs above threshold, yielding ER-like (random) topology — consistent with the "frozen" coordination pattern where novices rigidly couple all available DOFs.
- **Manuscript reference:** Fig 4 Panel A (left), §3.3, Table S8 Prediction P1.

### 2. `bassett_2011_nodes.csv` + `bassett_2011_edges.csv`
- **Source:** Bassett DS, Wymbs NF, Porter MA, Mucha PJ, Carlson JM, Grafton ST. Dynamic reconfiguration of human brain networks during learning. *Proc Natl Acad Sci USA*. 2011;108(18):7641–7646.
- **Content:** 12-node brain region network with 3 functional modules (Motor: M1/SMA/PMC/PMd; Sensory: S1/SPL/SMG/Vis; Cognitive: PFC/ACC/Ins/BG), intra-module edges (weight 0.75–0.85) and inter-module shortcut edges (weight 0.45).
- **Network interpretation:** Modular structure with sparse inter-module shortcuts yields WS-like (small-world) topology — consistent with the dynamic community structure Bassett et al. observed during motor sequence learning.
- **Manuscript reference:** Fig 4 Panel A (center), §3.3, Table S8 Prediction P2.

### 3. `scholz_schoner_1999_nodes.csv` + `scholz_schoner_1999_edges.csv`
- **Source:** Scholz JP, Schöner G. The uncontrolled manifold concept: identifying control variables for a functional task. *Exp Brain Res*. 1999;126(3):289–306.
- **Content:** 9-node joint hierarchy network for sit-to-stand task with CM (center of mass) as hub node connected to all peripheral DOFs (Ankle, Knee, Hip, Trunk, Shoulder, Head, θ-CM, ω-CM). Hub-to-peripheral weights range 0.70–0.95; peripheral-to-peripheral weights 0.35.
- **Network interpretation:** Star-like hub-dominated topology yields BA-like (scale-free) structure — consistent with UCM analysis showing V_UCM >> V_ORT, where the task-relevant variable (CM) acts as a network hub coordinating peripheral joint DOFs.
- **Manuscript reference:** Fig 4 Panel A (right), §3.3, Table S8 Prediction P3.

### 4. `kelso_1986_phase_transition.csv`
- **Source:** Kelso JAS, Scholz JP, Schöner G. Nonequilibrium phase transitions in coordinated biological motion: critical fluctuations. *Phys Lett A*. 1986;118(6):279–284.
- **Content:** 8 driving frequency conditions (1.25–3.00 Hz) with mean relative phase and SD for both anti-phase and in-phase bimanual coordination patterns. Documents the critical fluctuation peak (SD maximum) at the bifurcation point where anti-phase coordination becomes unstable and transitions to in-phase.
- **Network interpretation:** The SD peak at ~2.25 Hz maps to the percolation critical point where susceptibility χ(p) peaks — both represent maximum system sensitivity at a phase transition boundary.
- **Manuscript reference:** Fig 5 Panel E, §3.4, Correspondence C4.

### 5. `liu_2006_learning_parameters.json`
- **Source:** Liu Y-T, Mayer-Kress G, Newell KM. Qualitative and quantitative change in the dynamics of motor learning. *J Exp Psychol Hum Percept Perform*. 2006;32(2):380–393.
- **Content:** Piecewise function parameters for three-phase motor learning curve: plateau phase (t < 0.22), bifurcation/transition phase (0.22 ≤ t < 0.35), and stabilization phase (t ≥ 0.35). Includes variability envelope parameters showing CV peak during the transition phase.
- **Network interpretation:** The three-phase sigmoid maps to the coupled learning dynamics model (§2.4): subcritical plateau → percolation transition → supercritical refinement, with variability peak at the critical point corresponding to χ_max.
- **Manuscript reference:** Fig 5 Panel D, §3.4, Table 3 Prediction 8.

## Usage

All CSV files use standard comma-separated format with headers. The JSON file uses
standard JSON format. These files are loaded by the figure generation scripts in
`figures/generate_figures.py` and can be independently inspected for transparency.

```python
import pandas as pd
import json

# Load coupling matrix
coupling = pd.read_csv('data/empirical_reference/vereijken_1992_coupling.csv', index_col=0)

# Load brain network
nodes = pd.read_csv('data/empirical_reference/bassett_2011_nodes.csv')
edges = pd.read_csv('data/empirical_reference/bassett_2011_edges.csv')

# Load UCM hierarchy
ucm_nodes = pd.read_csv('data/empirical_reference/scholz_schoner_1999_nodes.csv')
ucm_edges = pd.read_csv('data/empirical_reference/scholz_schoner_1999_edges.csv')

# Load phase transition data
kelso = pd.read_csv('data/empirical_reference/kelso_1986_phase_transition.csv')

# Load learning parameters
with open('data/empirical_reference/liu_2006_learning_parameters.json') as f:
    liu_params = json.load(f)
```

## Citation

If using these structured data files, please cite both this repository and the
original source publications listed above.
