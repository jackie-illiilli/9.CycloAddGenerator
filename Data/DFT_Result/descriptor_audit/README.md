# Consolidated descriptor subset

The result is [Descriptors_Filter_merged.pkl](../Descriptors_Filter_merged.pkl).
The original `Descriptors_Filter.pkl`, the full local source in
`Ignored/Descriptors_.pkl`, and the modeling map in `Data/Descriptors` are unchanged.
Notebook paths have not been switched to the new file.

| Map | Molecules | Site records | Size (decimal MB) |
| --- | ---: | ---: | ---: |
| Full local source | 17,941 | 92,341 | 53.91 |
| Previous Filter | 2,329 | 12,728 | 7.42 |
| Consolidated subset | 5,325 | 32,797 | 19.26 |

The subset adds 2,996 molecules and 20,069 site records to the previous Filter's
coverage. Its file size is 64.3% smaller than the full source and below 50 MB.
It preserves the dictionary structure: keys are `Smiles_Id role atom_a atom_b`,
for example `02209 1 0 3`. Role `1` has 36 features (7,993 records); role `0`
has 28 features (24,804 records). All available roles and site records are kept
for each included molecule. Values and NumPy scalar types are not converted or
recalculated.

## Selection and merge policy

The scan inventories 265 repository files, including ignored local inputs.
Selection evidence consists of molecular Index columns in reaction tables and
candidate lists; existing descriptor subsets; explicit molecular constants and
saved text tables in notebooks; and molecular filenames and labeled text.
Selected SMILES lists and the paper's Excel cell values are also matched exactly
against the molecule catalog. Generic row indices, Reaction_Index, Structure_id,
conf_id and atom indices are not treated as molecule IDs.

The full descriptor source is not itself selection evidence. Molecules occurring
only in `all_smiles`, full FMO/Orbit tables or unselected `all_diene`/`all_ene`
pools are excluded. Candidate lists, including the raw 3,000-entry selections,
are retained. External paths are not followed. PDF, raster and editor binaries
are inventoried by filename but their contents are not OCR-scanned; available
source tables and notebook text provide the structured evidence for the figures.
See [inventory.json](inventory.json) for the per-file treatment and
[sources.json](sources.json) for the extracted IDs at each source location.

All previous Filter keys are retained. For overlapping keys, values come from
the **full `Ignored/Descriptors_.pkl`**, as requested. All 12,728 overlapping
records differ numerically from the previous Filter. Differences are principally
in the final eight area features; `18100 0 5 6` and `18100 0 6 5` also differ in
base features. The file therefore preserves the full source's descriptor version
throughout, rather than combining area features from different versions.
[conflicts.json](conflicts.json) records the differing zero-based feature
positions and both old and full-source values.

## Missing descriptors and validation

The scan requests 5,406 distinct indices. Of these, **81 have no descriptor in
either input**. All 81 occur in the raw candidate selections/selected SMILES
lists. They are reported and are not filled with invented values.

- [included_indices.txt](included_indices.txt): the 5,325 available molecule IDs.
- [requested_indices.txt](requested_indices.txt): all 5,406 requested IDs.
- [missing_indices.txt](missing_indices.txt): the 81 unavailable IDs.
- [molecules.json](molecules.json): per-index sources, record count and prior-Filter membership.
- [summary.json](summary.json): counts, input/output SHA-256 hashes and per-table site coverage.

Validation confirmed that every output vector equals its full-source vector,
every old Filter key survives, and all 8,637 entries in the existing modeling map
match the new subset exactly. All 32 CSV tables carrying Diene_Index, Ene_Index
and Title have complete required site-key coverage, with no unparsed rows.
Both Python 3.8 (`main`) and Python 3.12 (`main_py3_12`) can read the output with
ordinary `pickle.load`. Both source pickle hashes remain unchanged. No model
training or descriptor recalculation was performed.

## Rebuild

From the repository root, with the local full source available:

```powershell
conda run -n main python -B Code/filter_descriptors.py
```

The builder uses Python's standard library and NumPy. Build in the Python 3.8
`main` environment to serialize NumPy 1.x scalars with pickle protocol 4, keeping
the output readable in both project environments. It reads NumPy 2 scalar
namespaces in the SVO variant through a local unpickler compatibility mapping;
only that variant's molecule keys inform selection, not its feature values.
The builder refuses to produce a result if any inventoried input cannot be read.
