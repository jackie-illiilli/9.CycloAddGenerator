# Computational workflow for bioorthogonal [4+2] cycloaddition screeni

This repository contains notebooks, helper code, and selected data tables for the computational workflow described in the accompanying manuscript.

## Reproduction workflow

Start from the repository root and follow the stage guides in order:

1. Prepare the reactant pools with the selection workflow.
2. Generate or reuse quantum-chemical structures, energies, and descriptors.
3. Run the model-training and validation notebooks.
4. Apply the trained models and selection criteria to the reaction space.
5. Run the optional methanethiolate mechanism analysis when its external inputs are available.

The detailed notebook lists and required inputs are described in the guides for [reactant selection](notebooks/01_selection/README.md), [quantum calculations](notebooks/02_quantum/README.md), [modeling](notebooks/03_modeling/README.md), [discovery](notebooks/04_discovery/README.md), and [mechanism analysis](notebooks/05_mechanism/README.md). Set paths in each notebook's setup cell or through the documented environment variables. Quantum-chemistry notebooks require external job execution between input-generation and result-collection steps.

## Environment

The default environment is **`main_py3_12` (Python 3.12; locally 3.12.7)**.
[requirements.txt](requirements.txt) records the direct dependencies for this
workflow, including Hyperopt and the ChemProp/Lightning baseline. It is not a
complete environment lock file. For a new environment, use Python 3.12 and run
`python -m pip install -r requirements.txt`, followed by `python -m pip check`.
Select the `main_py3_12` interpreter/kernel when using the existing environment.

The dependency list is based on the local environment, with one compatibility
correction: NumPy is constrained to `>=1.26,<2.0` because the installed RDKit
2024.3.3 declares `numpy<2.0`. The inspected environment has NumPy 2.5.0 and
fails `pip check`; it also contains both Pandas 2.2.3 and 3.0.3 metadata, while
the imported version is 3.0.3. The requirements use that imported Pandas version.
These file updates do not repair or change the existing environment, and a fresh
installation of the complete dependency list has not yet been validated.
The local PyTorch build is `2.7.1+cu126`; the requirements pin the public version
`2.7.1`, so reproducing GPU behavior also requires selecting the appropriate CUDA
build and driver.

**Exception: running or testing
[03_fingerprint_generation.ipynb](notebooks/03_modeling/03_fingerprint_generation.ipynb)
must use `main` (Python 3.8; locally 3.8.18), not `main_py3_12`.** This project's
DScribe-based fingerprint workflow depends on the existing legacy environment.
Its installed descriptor stack includes `dscribe==1.2.1`, `ase==3.22.0`,
`mordred==1.2.0`, and `numpy==1.21.0`. Although `main_py3_12` has DScribe 2.1.1,
it is not the supported environment for this notebook. Keep the two environments
separate and do not install the Python 3.12 requirements into `main`.

For interactive execution, select the `main` Python 3.8 kernel in Jupyter or VS
Code; activating a terminal environment alone does not switch an open notebook's
kernel. If the kernel is not available, register it with:

```shell
conda run -n main python -m ipykernel install --user --name main --display-name "main (Python 3.8)"
```

Before running the fingerprint notebook, check `sys.executable` and
`sys.version` in its kernel to confirm that they point to `main` and Python 3.8.
Full execution additionally needs the external conformer files and pretrained
maps described in the [modeling guide](notebooks/03_modeling/README.md).

PDF export in `notebooks/04_discovery/06_draw_selected_reactions.ipynb` additionally
requires `reportlab` and `svglib` in its active kernel. Neither is installed in
the inspected `main_py3_12` environment, so their versions are not pinned here.
Gaussian and xTB/CREST remain external installations. Check each notebook's
setup cells for its required inputs and optional tools.

## Data availability and reproducibility limits

- The original ZINC download snapshot is not included. The saved reactant lists can be reused, but the original database query and exact selection cannot be reconstructed without the same source files.
- Selected energy and descriptor tables are provided, but the complete optimized-structure set and raw Gaussian/xTB logs are not. New DFT, transition-state, and IRC calculations cannot be reproduced from this repository alone.
- The 8,273 selected transition-state geometries are provided in [Data/DFT_Result/all_ts.xyz](Data/DFT_Result/all_ts.xyz), in the exact row order of Active_Energy_for_AttFP.csv. See the [DFT data guide](Data/DFT_Result/README.md) for identifiers, coordinate conventions, and the export command.
- The descriptor files in Data/Descriptors are filtered, precomputed subsets. The full source descriptor tables and their source calculation files are not included, so the subsets cannot be regenerated here.
- The full conformer collection and some pretrained molecular representations required by the fingerprint benchmark are external inputs.
- Full reaction-space prediction and the historical methanethiolate calculations require external descriptor maps, curated input tables, and completed calculation logs that are not bundled.
- Historical purchasability checks depend on supplier information from the time of screening; a dated supplier snapshot is not included.

Use the committed or otherwise supplied intermediate tables for the steps they support. Do not treat the repository as a complete archive of every calculation input or output.
