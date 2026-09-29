# Reactant-selection data

This directory contains input lists and saved intermediate tables for the reactant-selection workflow in notebooks/01_selection/1_select_4π_2π.ipynb. Here, diene denotes the 4π partner and ene denotes the 2π partner.

## File roles

- all_smiles.smi is the collected starting list used for structural filtering.
- all_diene.smi and all_ene.smi are the filtered 4π and 2π reactant pools.
- all_selected_diene.smi, all_selected_ene.smi, and all_selected_smiles.csv are the selected reactant lists.
- all_smiles.csv contains the available reactant calculation records.
- target_diene.csv and target_ene.csv contain reactants with usable saved calculation records; the corresponding raw files retain incomplete records.
- target_diene_ene.csv contains reference-partner reaction inputs, and target_diene_ene_Result.csv contains their saved calculation results.
- target_diene_passed.csv and target_ene_passed.csv are the saved reactant pools used by later reaction-pair generation.

These files are snapshots from different workflow stages. Preserve the reactant identifiers and schemas when joining them; rows in all_selected_smiles.csv are parallel reactant lists, not validated reaction pairs.

## Reproduction limits

The original downloaded ZINC .smi files are not included. The saved all_smiles.smi file is a collected input, not a complete ZINC release, and the exact source snapshot and query are not archived. Recreating the initial selection requires obtaining the original source files or an equivalent dated snapshot. Recreating the saved quantum-chemical screening results also requires the corresponding optimized structures and calculation outputs. Use the supplied intermediate tables for downstream steps when exact upstream inputs are unavailable.
