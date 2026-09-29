# Reactant selection

The selection notebook prepares the 4π and 2π reactant pools and the saved lists used by the quantum workflow.

Open the notebook from this repository or set CYCLOADD_REPO_ROOT to the repository directory. Configure the input and output locations in its setup cell, or set the supported environment variables: CYCLOADD_SMILES_DIR, CYCLOADD_PRED_CSV_DIR, CYCLOADD_REACTION_WORK_DIR, and CYCLOADD_REACTION_RESULT_CSV.

The original ZINC .smi download files are not included. To reproduce the initial selection, obtain the same source snapshot and place its files under Data/ZINC_Data/ZINC, then run the notebook with the matching calculation records. The saved filtered lists in Data/ZINC_Data can be reused for downstream work without repeating this step.
