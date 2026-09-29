# Quantum-chemistry workflow

The five notebooks separate reactant calculations, descriptor generation, reaction-pair preparation, reaction calculations, and selected comparison cases.

| Notebook | Task | Main inputs and outputs |
| --- | --- | --- |
| 01_reactant_dft.ipynb | Prepare and optimize reactant structures | Selected reactants; optimized structures and reactant result tables |
| 02_reactant_descriptors.ipynb | Calculate reactant descriptors | Reactant structures and calculation outputs; descriptor CSV and pickle files |
| 03_reaction_pair_generation.ipynb | Build reaction-pair and self-reaction inputs | Passed reactant lists, structures, and a descriptor map; reaction input tables |
| 04_reaction_dft_ts_irc.ipynb | Calculate reaction energies and optional TS/IRC results | Reaction inputs and reactant results; product, TS, IRC, and energy tables |
| 05_legacy_cases_and_dft_experiment.ipynb | Handle selected legacy cases and compare DFT with experiment | Curated case inputs and Data/DFT_Result/DFT_EXP.csv |

Run each notebook from its setup section and configure the external paths before generating inputs. The reaction-calculation notebook contains alternative conformer and calculation routes; choose the route needed for a job rather than running every branch. Gaussian and xTB/CREST are required where specified, with external calculations performed between the relevant notebook stages.

## Inputs not included

The original ZINC download, complete optimized-structure collection, and raw Gaussian/xTB job directories are not bundled. The reaction-pair generation workflow also expects the historical descriptor map all_property_detailSterimol_witharea.pkl, which is external and is not guaranteed to match Data/Descriptors/Descriptors_.pkl. Do not substitute the maps without checking their keys and feature definitions. The supplied tables support selected downstream analyses, but the full quantum workflow cannot be regenerated from this repository alone.
