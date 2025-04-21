# Selection Potential Bioorthogonal Reaction with Distortion/Interaction Method

This is a repository for paper  *""* . Here, you can find scripts used in this study.

# Requirement

```xml
ase==3.22.0
catboost==1.2
dscribe==1.2.1
matplotlib==3.3.4
mordred==1.2.0
morfeus-ml==0.7.2
numpy==1.21.0
pandas==2.0.3
rdkit-pypi==2022.9.4
scikit-learn==1.0
scipy==1.10.1
seaborn==0.12.2
sterimol==1.0
tensorflow==2.10.0
torch==2.0.0.dev20221221+cu117
tqdm==4.59.0
traitlets==5.2.2.post1
xgboost==1.6.1
```

# Usage

This project includes four core functionalities, each corresponding to one of the four `.ipynb` files. The four functionalities are as follows:

1. "Generation of D-A reactant datasets"
2. "Automated transition state structure construction and descriptor calculation"
3. "Training of machine learning models"
4. "Prediction in the D-A reaction chemical space and discovery of novel bioorthogonal reactions"

## 1. Generation of D-A reactants datasets

The `1_select_4π_2π.ipynb` file contains the process of generating a diverse D-A reaction dataset from the ZINC database through various screening methods, and it completes automated DFT calculations and descriptor computations. Before using it, please first download the required dataset from [https://zinc20.docking.org/](https://zinc20.docking.org/) and place it in the `Data/ZINC_Data/ZINC` directory. The automated DFT and descriptor calculations will call the code from `2_GaussianProcess.ipynb`.

## 2. Automated transition state structure construction and descriptor calculation

* In the section "DFT Calculate the Reactants" of this file, all SMILES strings listed under the "Diene" and "Ene" columns in the CSV files located in `csv_dirs` will be indexed and calculated. The structures of the reactants will be optimized through xTB and Gaussian optimization steps. Therefore, you can list the reactions to be calculated using the "Data/ZINC_Data/all_selected_smiles2.csv" as a template and then run the corresponding code. Whether constructing transition states on a large scale or predicting energies, this step is essential.
* In the section "Get Descriptors for Reactants," descriptors will be computed based on the Gaussian results and stored in the same folder as the Gaussian calculation files from the previous step.
* The section "Generate the All Cross-Combination Dataset with passed 4π and 2π" is designed to perform a full cross-combination of 4π and 2π components to generate a large reaction space. When using this, you can provide lists of molecular IDs for dienes and enes, corresponding to the molecules you want to combine.
* The section ""4. DFT Calculation Reactions" in this file will calculate the transition states of D-A reactions through a script. You need to modify 'reaction_file_dir' to the address of the CSV file that records the reactions to be calculated, and 'Result_csv' will be the CSV file where the results are saved. If you only need to obtain thermodynamic data, run the cell with the comment '# if Just Calculate deltaG, Run This and Ignore the Code After.' If you need to calculate D/I energy and kinetic energy, you should run all cells except for the one mentioned above. The results will include the activation electronic energy, activation free energy, D/I energy, and the activation free energy under water solvation

## 3. Training of machine learning models

The `3_model_train.ipynb` file documents the entire process required for training the machine learning model. The training requires PhysOrg descriptors, which depend on DFT-optimized geometric structures as well as Gaussian-calculated quantities such as charges and orbital energies. Therefore, it's necessary to first optimize the transition states of the reactants. Due to the large size of Gaussian output files, they are not provided on GitHub. If needed, please contact the email provided in the file 12237058@zju.edu.cn, and we can provide the Gaussian output files for all reactants and transition states.

To facilitate result reproduction, the file includes precomputed descriptor results for all reactants involved in the reaction space, saved as `Data\Iteration_Data\all_property_detailSterimol_witharea.pkl`. Therefore, to reproduce the model's performance, simply run this module. All the figures mentioned in the main text and SI of the paper can be found within the cells.

The pre-trained descriptors for deep learning used [NetPharMedGroup/publication_fingerprint: code for Zagidullin et al 2021 &#34;Comparative analysis of molecular fingerprints in prediction of drug combination effects&#34; (github.com)](https://github.com/NetPharMedGroup/publication_fingerprint), which can be downloaded from their corresponding GitHub repository. This project only provides the precomputed descriptors.

## 4. Prediction in the D-A reaction chemical space and discovery of novel bioorthogonal reactions

The `4_DI_discover.ipynb` file is used to predict D/I energy, thermodynamic energy, and kinetic energy using the model from `3_model_train.ipynb` on the fully cross-combined space generated by `2_GaussianProcess.ipynb`. The bioorthogonal reactions reported in the literature are sourced from Coelho et al. *J. Am. Chem. Soc*, **2020**, *142, 9*, 4235.

# How to cite

I don't know

# Contact with us

Email: 12237058@zju.edu.cn;

