# Logical Channel-Spectrum Benchmarking

Simulation and analysis code accompanying the manuscript **“Spectral Benchmarking of Logical Cycles for Fault-Tolerance Verification and Error Mitigation.”**

This repository studies channel-spectrum benchmarking (CSB) of logical quantum cycles using the Steane code, including fault-tolerance scaling, coherent-error characterization, logical Pauli-noise learning, and probabilistic error cancellation (PEC).

## Repository contents

| File | Purpose |
| --- | --- |
| `logical_csb.py` | Shared simulation and analysis routines. |
| `phase_gate_ft_scaling.ipynb` | Fault-tolerance scaling of the logical phase gate. |
| `phase_gate_learning_pec.ipynb` | Logical phase-gate noise learning and PEC. |
| `coherent_scaling.ipynb` | Analysis and plotting of coherent-error scaling. |
| `run_coherent_scaling_paper.py` | Runs coherent-error scaling simulations and saves results incrementally. |
| `coherent_phase_gate_scaling.sqlite` | Stored coherent-error scaling results. |
| `coherent_learning_pec.ipynb` | Noise learning and PEC in the presence of coherent errors. |
| `cx_ft_scaling_pec_model_comparison.ipynb` | Logical CX scaling, noise-model comparison, and PEC. |
| `logical_T_gate_CSB_learning_PEC.ipynb` | Logical T-gate CSB, noise learning, and PEC. |
| `universal_C_CX_C_T_PEC.ipynb` | PEC for composite logical circuits containing Clifford, CX, and T gates. |

## Setup

The main dependencies are NumPy, SciPy, pandas, Matplotlib, Numba, and joblib. JupyterLab is used to run the notebooks.
