# Filtered reactant descriptors

This directory contains precomputed descriptor files used by the modeling notebooks.

- Descriptors.csv contains the selected reactant/site records before the area features were added.
- Descriptors_.csv contains the corresponding records with area features.
- Descriptors_.pkl is the dictionary format used by the modeling notebooks.
- Descriptors_SVO_2.0.pickle is a separately generated SVO variant; it is not interchangeable with Descriptors_.pkl.

The filtered files were selected using the reactant indices in Data/ZINC_Data/target_diene_passed.csv and target_ene_passed.csv. Matching used reactant type and Smiles_Id, retained all site records for each selected reactant, and preserved the source rows. Descriptor values were not recalculated during filtering.

The complete source descriptor tables and the optimized structures and calculation outputs used to generate them are not included. The supplied subsets can be used for the modeling workflows, but the full descriptor collection cannot be regenerated from this repository alone. The original extraction directory was external to the repository, and no extraction script is included here.
