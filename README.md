# Computational workflow for bioorthogonal [4+2] cycloaddition screening

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

Use requirements.txt as a starting point for the Python environment. It is a legacy dependency list rather than a complete lock file. Some workflows require additional packages, including ChemProp and PyTorch Lightning, as well as external Gaussian and xTB/CREST installations. Check the setup cells for the notebook being used.

## Data availability and reproducibility limits

- The original ZINC download snapshot is not included. The saved reactant lists can be reused, but the original database query and exact selection cannot be reconstructed without the same source files.
- Selected energy and descriptor tables are provided, but the complete optimized-structure set and raw Gaussian/xTB logs are not. New DFT, transition-state, and IRC calculations cannot be reproduced from this repository alone.
- The descriptor files in Data/Descriptors are filtered, precomputed subsets. The full source descriptor tables and their source calculation files are not included, so the subsets cannot be regenerated here.
- The full conformer collection and some pretrained molecular representations required by the fingerprint benchmark are external inputs.
- Full reaction-space prediction and the historical methanethiolate calculations require external descriptor maps, curated input tables, and completed calculation logs that are not bundled.
- Historical purchasability checks depend on supplier information from the time of screening; a dated supplier snapshot is not included.

Use the committed or otherwise supplied intermediate tables for the steps they support. Do not treat the repository as a complete archive of every calculation input or output.
