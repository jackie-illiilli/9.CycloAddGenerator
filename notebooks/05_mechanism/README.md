# Methanethiolate mechanism workflow

These notebooks prepare inputs, collect completed quantum-chemistry jobs, and analyze the saved energies and geometries.

| Notebook | Task |
| --- | --- |
| 01_mes_product_inputs.ipynb | Prepare product structures and calculation inputs |
| 02_mes_ts_irc_and_spe.ipynb | Prepare TS, IRC, and single-point jobs and select completed records |
| 03_mes_thermodynamics_and_kinetics.ipynb | Collect thermodynamic and kinetic quantities |
| 04_mes_distortion_and_correlations.ipynb | Prepare and analyze distortion calculations |
| 05_da_mes_bond_length_analysis.ipynb | Compare bond-length changes in selected structures |
| 06_historical_eight_point_correlation.ipynb | Historical manual analysis; separate from the computed workflow |

Run the notebooks in sequence where one stage consumes the previous stage's files. Configure paths in notebooks/_workflow_paths.py or through its environment variables. External Gaussian jobs must be completed before the result-collection cells are run.

## Inputs not included

The curated MeS reaction input table, complete optimized structures, and generated Gaussian product, TS, IRC, and single-point logs are not included. Without those files, the quantum calculations and their derived mechanism analyses cannot be regenerated. The notebooks can prepare some inputs and analyze supplied outputs, but they do not replace the missing calculation archive.
