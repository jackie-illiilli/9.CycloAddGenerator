# DFT results and transition-state coordinates

`all_ts.xyz` contains 8,273 transition-state geometries as concatenated XYZ
frames. Frame 1 corresponds to the first data row of
`Active_Energy_for_AttFP.csv`; every subsequent frame follows the CSV row order,
without sorting or deduplication.

Each frame contains the atom count, a comment line with the source identifier
`{Diene_Index:05d}_{Ene_Index:05d}_{Structure_id:05d}_{conf_id:04d}`, and element
symbols with Cartesian coordinates in angstroms. Atom order and coordinates are
taken from the final **Standard orientation** table in the corresponding Gaussian
log. These coordinates are the final reported geometries, not the initial input
structures or imaginary-mode displacements.

The source logs are external, originally located at
`E:\work\Secondary_Selection\ZINC_1\ts`. Rebuild from the repository root with:

```powershell
python Code/export_ts_xyz.py --logs E:\work\Secondary_Selection\ZINC_1\ts
```

The exporter uses only the Python standard library. It checks for missing logs,
normal termination, optimization completion, and valid coordinate tables, and
replaces the output only after all rows have been exported successfully.

The XYZ archive supplies the selected TS coordinates for this table; it does not
include the full conformer collection, reactant/product structures, vibrational
modes, or raw calculation logs.
