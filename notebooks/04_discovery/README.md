# Discovery workflow

Run the notebooks in order. Each notebook locates the repository root and loads its own inputs; configure external paths in the setup cells or through notebooks/_workflow_paths.py.

| Notebook | Task |
| --- | --- |
| 01_full_space_prediction.ipynb | Prepare descriptors and predict the reaction space |
| 02_energy_and_self_reaction_filter.ipynb | Apply energetic and curated self-reaction filters |
| 03_iedda_and_distortion_selection.ipynb | Classify IEDDA candidates, compare distortion and interaction distributions, and apply the distortion-fraction selection |
| 04_reported_bo_reference.ipynb | Compare candidates with the curated reported-reaction reference set |
| 05_purchasability_and_dft_validation.ipynb | Review candidate availability and compare selected predictions with DFT records |

The selection notebook uses the curated reference data in Data/Reported_BO. Keep the intermediate tables and reactant identifiers aligned across stages.

## Inputs not included

The full reaction-space descriptor map and prediction table are external and are not bundled. The workflow can use supplied downstream snapshots where available, but the full-space predictions cannot be regenerated without the corresponding structures and descriptor data. The historical supplier catalog snapshot and complete raw DFT job outputs are also unavailable, so past availability checks and new candidate calculations cannot be independently reproduced exactly.
