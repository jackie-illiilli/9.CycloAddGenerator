import os, glob, math, shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from . import logfile_process, format_change, xtb_process, cycle_process, Tool
from tqdm import tqdm 

ALL_PROPERTIES_diene = ["diene_charge_a", "diene_charge_b", "diene_charge_c", "diene_charge_d",
    "diene_max_neighbor_charge", "diene_min_neighbor_charge", "diene_max_charge", "diene_min_charge", "bond_1", "bond2", "bond3", "distance_ad",
    "angle_a_cos", "angle_b_cos", "diene_max_angle_cos", "diene_min_angle_cos", "torsion",
    "diene_homo-1", "diene_homo", "diene_lumo", "diene_lumo+1", "diene_dipole", 
    "max_diene_L", "min_diene_L", "max_diene_B1", "min_diene_B1", "max_diene_B5", "min_diene_B5"]
ALL_PROPERTIES_ene = ["ene_charge_e", "ene_charge_f", "ene_max_neighbor_charge", "ene_min_neighbor_charge", 
    "diene_max_charge", "diene_min_charge", "bond4", 
    "ene_max_angle_cos", "ene_min_angle_cos", "ene_homo-1", "ene_homo", "ene_lumo", "ene_lumo+1", 
    "ene_dipole", "max_ene_L", "min_ene_L", "max_ene_B1", "min_ene_B1", "max_ene_B5", "min_ene_B5"]
new_des = ['diene_area_0', 'diene_area_1', 'diene_area_2', 'diene_area_3', 'diene_area_4', 'diene_area_5', 'diene_area_6', 'diene_area_7', 
'ene_area_0', 'ene_area_1', 'ene_area_2', 'ene_area_3', 'ene_area_4', 'ene_area_5', 'ene_area_6', 'ene_area_7', ]
AtomsIds = ["H", "C", "N", "O"]
    
DIENE_ENE_DIR = r"G:\work\Secondary_Selection\Diene_Ene_Smiles"
# DIENE_ENE_DIR = r"G:\work\First_calculation\Diene_Ene_Smiles"


def read_smiles_in_csv(csv_dirs=["G:/work/Calced_Data/result_sum_reaction.csv"]):
    all_smiles_lists = np.array([])
    for idx, csv_dir in enumerate(csv_dirs):
        smiles_lists = np.array([])
        print(idx, len(csv_dirs), len(all_smiles_lists), end='\r')
        csv = pd.read_csv(csv_dir)
        for each_column in csv.keys():
            try:
                assert Chem.MolFromSmiles(csv[each_column][0]) != None
            except:
                continue
            smiles_list = csv[each_column].to_numpy()
            smiles_lists = np.concatenate((smiles_lists, smiles_list))
        smiles_lists = np.unique(smiles_lists)
        all_smiles_lists = np.concatenate((all_smiles_lists, smiles_lists))
    all_smiles_lists = np.unique(all_smiles_lists)
    return all_smiles_lists

def collect_smiles_file(csv_dirs, root_dir, sugan=True, old_smiles_csv = None):  
    target_dir = root_dir
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    all_smiles = read_smiles_in_csv(csv_dirs)
    all_smiles.sort()
    if old_smiles_csv != None:
        old_smiles = pd.read_csv(old_smiles_csv)["Smiles"].to_numpy()
    else: old_smiles = np.array([])
    all_smiles = np.setdiff1d(all_smiles, old_smiles)
    format_change.write_smi_csv(all_smiles, target_dir + "/all_smiles.csv", start_id=len(old_smiles))
    molfile_dir =  target_dir + "/mol"
    if not os.path.isdir(molfile_dir):
        os.mkdir(molfile_dir)
    all_mols, all_mol_names = [], []
    for smiles_id, smiles in enumerate(all_smiles):
        try:
            mol = cycle_process.to_trans_cycloene(smiles, conf_num=1)
        except:
            print(smiles)
            continue
        if len(mol.GetConformers()) == 0:
            print("%s is not well" % smiles)

        all_mols.append(mol)
        mol_name = "smilesid_%.5d" % (smiles_id + len(old_smiles))
        all_mol_names.append(mol_name)
        Chem.MolToMolFile(mol, molfile_dir + "/" + "%s.mol" % mol_name)
    xtb_process.xtb_main(all_mol_names, all_mols, dir_path=target_dir + '/' + 'first_xtb', que="gamma", core=30)
    if sugan:
        pbs_files = glob.glob(target_dir + '/' + 'first_xtb/*.pbs')
        for pbs_file in pbs_files:
            with open(pbs_file, "wt",  newline='\n') as f:
                number = int(pbs_file.split(".pbs")[0].split("_")[-1])
                f.write("#!/bin/bash\n#SBATCH -J g16\n#SBATCH -N 1\n#SBATCH --ntasks-per-node=32\n#SBATCH -p hfacnormal01\n\nroot=`pwd`\nrootdir=charg_0_%d\nfolders=`ls $root/$rootdir/`\n" % number)
                f.write("for folder in $folders\ndo\n    cd $root/$rootdir/$folder\n    crest $folder.xyz -T 28 -gfn2 -chrg 0 -uhf 0 -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\ndone\n")
        with open(target_dir + '/' + 'first_xtb/suball', "wt",  newline='\n') as f:
            for pbs_file in pbs_files:
                name = os.path.split(pbs_file)[-1]
                f.write("sbatch %s\n" % name)

def read_title(target_dir):
    title_file = target_dir + '/' + 'xtb_title.txt'
    title_dict = {}
    with open(title_file, 'rt') as f:
        lines = f.readlines()
    for eachline in lines:
        eachline = eachline.strip("\n")
        mol_name = str(eachline.split("#")[0])
        title = eachline.split("#")[1]
        title_dict[mol_name] = title
    return title_dict

def smiles_DFT_calc(target_dir, 
                    has_title=None, 
                    root_dir='first_xtb', 
                    mol_dir='mol', 
                    dft_dir='mol_dft', 
                    method="opt freq b3lyp/6-31g* em=gd3bj",
                    conf_limit=3,
                    rmsd_limit=1.5,
                    freeze=False,
                    ):
    if has_title != None:
        title_dict = read_title(target_dir)
    root_dir= target_dir + '/' + root_dir
    mol_dir = target_dir + '/' + mol_dir
    dft_dir = target_dir + '/' + dft_dir
    all_files = glob.glob(root_dir + "/*/*/*")
    for xtb_file in all_files:
        if ("crest.out" in xtb_file) or ("best" in xtb_file) or ("crest_conf" in xtb_file):
            pass
        else:
            if os.path.isdir(xtb_file):
                shutil.rmtree(xtb_file)
            else:
                os.remove((xtb_file))
    mol_files = glob.glob(mol_dir + "/*.mol")
    for i, mol_file in enumerate(mol_files):
        mol_name = os.path.split(mol_file)[-1].split(".")[0]
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        if has_title == None:
            title = "Singlemol"
        else:
            try:
                title = title_dict[mol_name]
            except:
                print("%s not in title" % mol_name)
                continue
        if freeze: 
            title_num = [int(each) for each in title.split()]
            freeze_dict = [[title_num[0] + 1, title_num[2] + title_num[4] + 1], [title_num[1] + 1, title_num[3] + title_num[4] + 1]]
            xtb_process.after_xtb(mol,root_dir=root_dir + "/*", save_dir=dft_dir, mol_str=mol_name, xtb_title=title, method=method, conf_limit=conf_limit, rmsd_limit=rmsd_limit, freeze=freeze_dict)
        else:
            xtb_process.after_xtb(mol,root_dir=root_dir + "/*", save_dir=dft_dir, mol_str=mol_name, xtb_title=title, method=method, conf_limit=conf_limit, rmsd_limit=rmsd_limit)

def read_engs(opt_log_files, eng_dir, returnE=False):
    all_engs = []
    all_conf_id = []
    all_E_engs = []
    for opt_log_file in opt_log_files:
        eng_log_files = glob.glob(eng_dir + "/" + os.path.split(opt_log_file)[1])
        if len(eng_log_files) == 0:
            continue
        assert len(eng_log_files) == 1
        eng_log_file = eng_log_files[0]
        try:
            conf_id = int(eng_log_file.split("_")[-1].split(".")[0])
        except:
            conf_id = 0
        opt_log = logfile_process.Logfile(opt_log_file)
        eng_log = logfile_process.Logfile(eng_log_file)
        assert len(opt_log.all_engs) == 5
        opt_G_cor = opt_log.all_engs[-1]
        eng_SPE = eng_log.all_engs[0]
        all_engs.append(opt_G_cor + eng_SPE)
        all_E_engs.append(eng_SPE)
        all_conf_id.append(conf_id)
    if returnE:
        return all_engs, all_conf_id, all_E_engs
    return all_engs, all_conf_id

def smiles_result_analysis(target_dir):
    opt_file_dir = target_dir + "/mol_dft"
    eng_dir = target_dir + "/mol_dft_eng"
    solvent_eng_dir = target_dir + "/mol_dft_eng_solvent"
    smiles_dict = pd.read_csv(target_dir + "/" + "all_smiles.csv").to_dict()
    smiles_dict["G/Hatree"] = {}
    smiles_dict["G(Solvent)/Hatree"] = {}
    smiles_dict["E/Hatree"] = {}
    smiles_dict["Stable_conf_id"] = {}
    for idx, _ in tqdm(enumerate(list(smiles_dict["Smiles"]))):
        smiles_id = smiles_dict["Index"][idx]
        mol_name = "smilesid_%.5d" % smiles_id
        mol_file = glob.glob(target_dir + "/mol/%s.mol" % mol_name) 
        if len(mol_file) == 0:
            smiles_dict["G/Hatree"][idx] = "RDKIT Generate Fail"
            smiles_dict["Stable_conf_id"][idx] = -1
            continue
        opt_log_files = glob.glob(opt_file_dir + "/" + mol_name + "*.log")
        if len(opt_log_files) == 0:
            smiles_dict["G/Hatree"][idx] = "DFT OPT Fail"
            smiles_dict["Stable_conf_id"][idx] = -1
            continue 
        # 
        all_engs, all_conf_id, all_E_engs = read_engs(opt_log_files, eng_dir, returnE=1)
        if len(all_engs) == 0:
            smiles_dict["G/Hatree"][idx] = "DFT ENG Fail"
            smiles_dict["Stable_conf_id"][idx] = -1
            continue
        min_conf_idx = np.argmin(all_engs)
        min_engs = all_engs[min_conf_idx]
        min_E_engs = all_E_engs[min_conf_idx]
        min_conf_idx = all_conf_id[min_conf_idx]
        smiles_dict["G/Hatree"][idx] = min_engs
        smiles_dict["E/Hatree"][idx] = min_E_engs
        smiles_dict["Stable_conf_id"][idx] = min_conf_idx
        # Solvent
        all_engs, all_conf_id, all_E_engs = read_engs(opt_log_files, solvent_eng_dir, returnE=1)
        if len(all_engs) == 0:
            smiles_dict["G(Solvent)/Hatree"][idx] = "DFT ENG Fail"
            smiles_dict["Stable_conf_id"][idx] = -1
            continue
        min_conf_idx = np.argmin(all_engs)
        min_engs = all_engs[min_conf_idx]
        min_conf_idx = all_conf_id[min_conf_idx]
        smiles_dict["G(Solvent)/Hatree"][idx] = min_engs
        smiles_dict["Stable_conf_id"][idx] = min_conf_idx

        # mol = Chem.MolFromMolFile(mol_file[0], removeHs=False)
        # new_mol_name = "smilesid_%.5d_%.4d" % (smiles_id, min_conf_idx)
        # mol_log_files = glob.glob(opt_file_dir + "/" + new_mol_name + "*.log")
        # assert len(mol_log_files) == 1
        # mol_log = logfile_process.Logfile(mol_log_files[0])
        # atom_list = mol_log.symbol_list
        # position = mol_log.first_atom_position
        # mol = xtb_process.xtb_to_mol(mol, [atom_list], [position], 1)
        # Chem.MolToMolFile(mol, mol_file[0])

    smiles_dict = pd.DataFrame(smiles_dict)
    smiles_dict.to_csv(target_dir + "/" + "all_smiles.csv", index=False)

def update_log_to_mol(target_dir):
    smiles_csv = pd.read_csv(target_dir + "/all_smiles.csv")
    new_mol_dir = target_dir + "/new_mols/"
    if not os.path.isdir(new_mol_dir):
        os.mkdir(new_mol_dir)
    for idx, smiles_index in enumerate(smiles_csv['Index']):
        conf_id = smiles_csv['Stable_conf_id'][idx] 
        mol_file = glob.glob(target_dir + "/mol/smilesid_%.5d.mol" % smiles_index)
        if len(mol_file) == 0:
            continue
        mol_file = mol_file[0]
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        log_files = glob.glob(target_dir + '/mol_dft/smilesid_%.5d_%.4d.log'% (smiles_index, conf_id))
        if len(log_files) == 0: 
            continue
        log_file = log_files[0]
        opt_log = logfile_process.Logfile(log_file)
        position = opt_log.running_positions[-1]
        atom_lists = opt_log.symbol_list
        new_mol = xtb_process.xtb_to_mol(mol, [atom_lists], [position], conf_limit=1)
        Chem.MolToMolFile(new_mol, new_mol_dir +'smilesid_%.5d.mol' % smiles_index)


def smiles_property_generate(appeared_idxs = [], file_name = 'property.csv', smiles_dict=pd.read_csv(r"G:\work\Secondary_Selection\Diene_Ene_Smiles\all_smiles.csv"), smiles_dir=r"G:\work\Secondary_Selection\Diene_Ene_Smiles"):
    target_dict = {"Smiles":{}, "Smiles_Id":{}, "Type":{}, "Title":{}}
    for each in ALL_PROPERTIES_diene:
        
        target_dict[each] = {}
    for each in ALL_PROPERTIES_ene:
        target_dict[each] = {}
    target_dict_id = 0
    for idx in tqdm(range(len(smiles_dict["Smiles"]))):
        smiles_id = smiles_dict["Index"][idx]
        smiles = smiles_dict["Smiles"][idx]
        conf_id = smiles_dict["Stable_conf_id"][idx]
        
        if smiles_id not in appeared_idxs and len(appeared_idxs) > 0:
            continue
        if conf_id < 0:
            continue
        
        # print(smiles_id, conf_id)
        mol_dir = smiles_dir + "/mol/smilesid_%.5d.mol" % smiles_id
        mol = Chem.MolFromMolFile(mol_dir, removeHs=False)
        log_dir = smiles_dir + "/mol_dft/smilesid_%.5d_%.4d.log" % (smiles_id, conf_id)
        log = logfile_process.Logfile(log_dir)

        diene_atom_lists = cycle_process.diene_atom_Idx(mol, select_diene=0)
        for diene_atom_list in diene_atom_lists:
            try:
                atoma_id, atomb_id, atomc_id, atomd_id = diene_atom_list
                atom_symbol = [mol.GetAtomWithIdx(each).GetSymbol() for each in diene_atom_list]
                # if atom_symbol[0] != "C" or atom_symbol[-1] != "C":
                #     continue
                # if atom_symbol[1] != "N" or atom_symbol[2] != "N":
                #     continue
                diene_neighbor = [[each.GetIdx() for each in mol.GetAtomWithIdx(diene_atom).GetNeighbors() if each.GetIdx() not in diene_atom_list] for diene_atom in diene_atom_list]
                # diene_property:
                require_property = []
                diene_charges = log.read_charge()
                require_property += [diene_charges[each] for each in diene_atom_list]
                diene_neighbor_charges = [[diene_charges[each] for each in each_]for each_ in diene_neighbor]
                diene_neighbor_charges = diene_neighbor_charges[0] + diene_neighbor_charges[1] + diene_neighbor_charges[2] + diene_neighbor_charges[3]
                if len(diene_neighbor_charges) == 0: diene_neighbor_charges = [0]
                require_property += [max(diene_neighbor_charges), min(diene_neighbor_charges)]

                diene_other_charges = [each for i, each in enumerate(diene_charges) if i not in diene_atom_list]
                if len(diene_other_charges) == 0: diene_other_charges = [0,0]
                require_property += [max(diene_other_charges), min(diene_other_charges)]

                diene_position = log.running_positions[-1]
                bond1 = Tool.get_atoms_distance(diene_position[atoma_id], diene_position[atomb_id])
                bond2 = Tool.get_atoms_distance(diene_position[atomb_id], diene_position[atomc_id])
                bond3 = Tool.get_atoms_distance(diene_position[atomc_id], diene_position[atomd_id])
                distance_ad = Tool.get_atoms_distance(diene_position[atoma_id], diene_position[atomd_id])
                require_property += [bond1, bond2, bond3, distance_ad]
                
                angle_a = Tool.get_bond_angle_deg(diene_position[atoma_id], diene_position[atomb_id], diene_position[atomc_id])
                angle_b = Tool.get_bond_angle_deg(diene_position[atomb_id], diene_position[atomc_id], diene_position[atomd_id])
                a_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atoma_id).GetNeighbors() if each.GetIdx() != atomb_id]
                d_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atomd_id).GetNeighbors() if each.GetIdx() != atomc_id]
                diene_angles = [Tool.get_bond_angle_deg(diene_position[each], diene_position[atoma_id], diene_position[atomb_id]) for each in a_neighbor] + \
                [Tool.get_bond_angle_deg(diene_position[atomc_id], diene_position[atomd_id], diene_position[each]) for each in d_neighbor]
                if len(a_neighbor) + len(d_neighbor) == 0:diene_angles = [0, 0]
                require_property += [angle_a, angle_b, min(diene_angles), max(diene_angles)]
                torsion = Tool.get_torsion_(diene_position[atoma_id], diene_position[atomb_id], diene_position[atomc_id], diene_position[atomd_id])
                require_property += [torsion]
                diene_orbit_eng = log.read_orbit_eng(HOMO_index=[-2, -1], LUMO_index=[0,1])
                diene_dipole = log.get_dipole()
                require_property += diene_orbit_eng + [diene_dipole]

                diene_L, diene_B1, diene_B5= [], [], []
                for each in a_neighbor:
                    L, B1, B5 = log.get_sterimol_parameters(atoma_id, each)
                    diene_L.append(L); diene_B1.append(B1), diene_B5.append(B5)
                for each in d_neighbor:
                    L, B1, B5 = log.get_sterimol_parameters(atomd_id, each)
                    diene_L.append(L); diene_B1.append(B1), diene_B5.append(B5)
                if len(diene_L) == 0: diene_L = [0]
                if len(diene_B1) == 0: diene_B1 = [0]
                if len(diene_B5) == 0: diene_B5 = [0]
                require_property += [max(diene_L), min(diene_L), max(diene_B1), min(diene_B1), max(diene_B5), min(diene_B5)]

                assert len(ALL_PROPERTIES_diene) == len(require_property)
                target_dict["Smiles"][target_dict_id] = smiles
                target_dict["Smiles_Id"][target_dict_id] = smiles_id
                target_dict["Type"][target_dict_id] = "Diene"
                target_dict["Title"][target_dict_id] = "%d %d" % (atoma_id, atomd_id)
                for each, value in zip(ALL_PROPERTIES_diene, require_property):
                    target_dict[each][target_dict_id] = value
                target_dict_id += 1
            except:
                continue

        ene_atom_lists = cycle_process.dieno_atom_Idx(mol)
        for each_ene_atom_list in ene_atom_lists:
            # if mol.GetAtomWithIdx(each_ene_atom_list[0]).GetIsAromatic() or mol.GetAtomWithIdx(each_ene_atom_list[-1]).GetIsAromatic():
            #     continue
            for ene_atom_list in [each_ene_atom_list, each_ene_atom_list[::-1]]:
                try:
                    ene_neighbor = [[each.GetIdx() for each in mol.GetAtomWithIdx(ene_atom).GetNeighbors() if each.GetIdx() not in ene_atom_list] for ene_atom in ene_atom_list]
                    require_property = []
                    atome_id, atomf_id = ene_atom_list
                    ene_charges = log.read_charge()

                    require_property += [ene_charges[each] for each in [atome_id, atomf_id]]
                    ene_neighbor_charges = [[ene_charges[each] for each in each_] for each_ in ene_neighbor]
                    ene_neighbor_charges = ene_neighbor_charges[0] + ene_neighbor_charges[1]
                    if len(ene_neighbor_charges) == 0: ene_neighbor_charges = [0]
                    require_property += [max(ene_neighbor_charges), min(ene_neighbor_charges)]
                    ene_other_charges = [each for i, each in enumerate(ene_charges) if i not in [atome_id, atomf_id]]
                    if len(ene_other_charges) == 0: ene_other_charges = [0,0]
                    require_property += [max(ene_other_charges), min(ene_other_charges)]

                    ene_position = log.running_positions[-1]
                    bond4 = Tool.get_atoms_distance(ene_position[atome_id], ene_position[atomf_id])
                    require_property += [bond4]

                    e_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atome_id).GetNeighbors() if each.GetIdx() != atomf_id]
                    f_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atomf_id).GetNeighbors() if each.GetIdx() != atome_id]
                    ene_angles = [Tool.get_bond_angle_deg(ene_position[each], ene_position[atome_id], ene_position[atomf_id]) for each in e_neighbor] + \
                    [Tool.get_bond_angle_deg(ene_position[atome_id], ene_position[atomf_id], ene_position[each]) for each in f_neighbor]
                    if len(e_neighbor) + len(f_neighbor) == 0:ene_angles = [0, 0]
                    require_property += [ min(ene_angles), max(ene_angles)]
                    try:
                        ene_orbit_eng = log.read_orbit_eng(HOMO_index=[-2, -1], LUMO_index=[0,1])
                        ene_dipole = log.get_dipole()
                        require_property += ene_orbit_eng + [ene_dipole]
                    except:
                        continue

                    ene_L, ene_B1, ene_B5 = [], [], []
                    for each in e_neighbor:
                        L, B1, B5 = log.get_sterimol_parameters(atome_id, each)
                        ene_L.append(L); ene_B1.append(B1), ene_B5.append(B5)
                    for each in range(2 - len(e_neighbor)):
                        ene_L.append(0); ene_B1.append(0), ene_B5.append(0)
                    for each in f_neighbor:
                        L, B1, B5 = log.get_sterimol_parameters(atomf_id, each)
                        ene_L.append(L); ene_B1.append(B1), ene_B5.append(B5)
                    for each in range(2 - len(f_neighbor)):
                        ene_L.append(0); ene_B1.append(0), ene_B5.append(0)
                    if len(ene_L) == 0: ene_L = [0]
                    if len(ene_B1) == 0: ene_B1 = [0]
                    if len(ene_B5) == 0: ene_B5 = [0]
                    require_property += [max(ene_L), min(ene_L), max(ene_B1), min(ene_B1), max(ene_B5), min(ene_B5)]

                    assert len(ALL_PROPERTIES_ene) == len(require_property)
                    target_dict["Smiles"][target_dict_id] = smiles
                    target_dict["Smiles_Id"][target_dict_id] = smiles_id
                    target_dict["Type"][target_dict_id] = "Ene"
                    target_dict["Title"][target_dict_id] = "%d %d" % (atome_id, atomf_id)
                    for each, value in zip(ALL_PROPERTIES_ene, require_property):
                        target_dict[each][target_dict_id] = value
                    target_dict_id += 1
                except:
                    continue
    a = pd.DataFrame(target_dict)
    a.to_csv(smiles_dir + "/" + file_name)

def SPE_DFT_calc(target_dir, opt_name = "mol_dft", eng_name = "mol_dft_eng", save_chk=None, method="b3lyp/6-311+g(d,p) em=gd3bj"):
    opt_file_dir = target_dir + "/" + opt_name
    eng_dir = target_dir + "/" + eng_name
    mol_files = glob.glob(target_dir + "/mol/*.mol")
    for mol_file in tqdm(mol_files):
        log_files = glob.glob(opt_file_dir + "/" + os.path.split(mol_file)[-1].split(".")[0] + "*.log")
        if len(log_files) == 0:
            continue 
        for log_file in log_files:
            new_log_name = eng_dir + "/" + os.path.split(log_file)[-1].split('.')[0] + ".gjf" 
            opt_log = logfile_process.Logfile(log_file, mol_file_dir=mol_file)
            assert len(opt_log.running_positions) != 0
            title, charge, symbol_list, position,= opt_log.title, opt_log.charge, opt_log.symbol_list, opt_log.running_positions[-1]
            title = " ".join(str(each) for each in title)
            if save_chk:
                savechk = os.path.split(new_log_name.strip(".gjf"))[-1]
            else:
                savechk = None
            format_change.block_to_gjf(symbol_list, position, new_log_name, charge, title,
                        method=method, savechk=savechk)

def SPE_DFT_calc_m06(target_dir):
    smiles_dict = pd.read_csv(r"D:\work\Data\cycle_addition\Diene_Ene_Smiles\all_smiles.csv")
    opt_file_dir = target_dir + "/mol_dft"
    eng_dir = target_dir + "/mol_dft_eng_m06"
    for idx in range(len(smiles_dict["Smiles"])):
        smiles_id = smiles_dict["Index"][idx]
        conf_id = smiles_dict["Stable_conf_id"][idx]
        if conf_id < 0:
            continue
        mol_file = target_dir + "/mol/smilesid_%.5d.mol" % smiles_id
        log_files = glob.glob(opt_file_dir + "/" + "smilesid_%.5d_%.4d.log" % (smiles_id, conf_id))
        if len(log_files) == 0:
            raise TypeError("Not such logfiles %d" % smiles_id) 
        for log_file in log_files:
            new_log_name = eng_dir + "/" + os.path.split(log_file)[-1].split('.')[0] + ".gjf"
            wfn_name = os.path.split(log_file)[-1].split('.')[0] + ".wfn"
            opt_log = logfile_process.Logfile(log_file, mol_file_dir=mol_file)
            assert len(opt_log.running_positions) != 0
            title, charge, symbol_list, position,= opt_log.title, opt_log.charge, opt_log.symbol_list, opt_log.running_positions[-1]
            title = " ".join(str(each) for each in title)
            format_change.block_to_gjf(symbol_list, position, new_log_name, charge, title,
                        method="m062x 6-311+g(d) scrf=(CPCM, solvent=dichloromethane) out=wfn", final_line=wfn_name)

def error_improve(target_dir, file_name, dust_bin='dust_bin', improve_name='improve'):
    opt_file_dir = target_dir + "/" + file_name
    dust_bin_dir = target_dir + "/" + dust_bin
    mol_files = glob.glob(target_dir + "/mol/*.mol")
    
    for mol_file in tqdm(mol_files):
        # print("process %s" % mol_file, end='\r')
        log_files = glob.glob(opt_file_dir + "/" + os.path.split(mol_file)[-1].split(".")[0] + "*.log")
        if len(log_files) == 0:
            continue 
        for log_file in log_files:
            fail = 0
            try:
                opt_log= logfile_process.Logfile(log_file, mol_file_dir=mol_file)
                file_type = opt_log.file_type
                if opt_log.multiplicity <= 0:
                    fail = 1
                elif not opt_log.bond_attach:
                    fail = 1
                if opt_log.file_type == "OM" and opt_log.unreal_freq == 0:
                    print("%s may not be a right OM for unreal freq num of %d" % (opt_log.file_dir, opt_log.unreal_freq))
                if opt_log.file_type == "TS" and opt_log.unreal_freq == 0:
                    print("%s is not a right TS for unreal freq num of %d" % (opt_log.file_dir, opt_log.unreal_freq))
                    fail = 1
                if opt_log.file_type == "IRC":
                    if opt_log.irc_result == False:
                        fail = 1
                    else:
                        continue
            except:
                fail = 1
                file_type = "ERROR"
                
            if fail == 1:
                new_log_name = dust_bin_dir + "/" + os.path.split(log_file)[-1] 
                new_log_name =  new_log_name.split(".")[0] + "%s.log" % file_type
                if not os.path.isdir(dust_bin_dir):
                    os.mkdir(dust_bin_dir) 
                shutil.move(log_file, new_log_name)
                continue
            new_log_name = target_dir + "/" + improve_name 
            savechk = None
            readchk = None
            # if opt_log.file_type == "OM":
            #     savechk = os.path.split(log_file)[-1].split(".")[0]
            # if opt_log.file_type == "TS":
            #     readchk = os.path.split(log_file)[-1].split(".")[0]
            if not opt_log.normal_end:
                opt_log.solve_error_logfile(new_log_name, savechk=savechk, readchk=readchk)
            elif opt_log.unreal_freq and opt_log.file_type not in ["OM", "TS"]:
                opt_log.unreal_freq_improve(new_log_name, savechk=savechk, readchk=readchk)

def collect_std_Hirshfeld_chg(key_dir, return_file="React_atom2.csv", reading_method=format_change.read_chg_file):
    
    smiles_dict = pd.read_csv(key_dir + "/React_atom.csv").to_dict()
    smiles_dict["C_max"], smiles_dict["C_min"],smiles_dict["N_max"], smiles_dict["N_min"] = {}, {}, {}, {}
    smiles_dict["O_max"], smiles_dict["O_min"],smiles_dict["H_max"], smiles_dict["H_min"] = {}, {}, {}, {}
    smiles_dict["charge_select"], smiles_dict["charge_giveup"] = {}, {}
    for id, name in enumerate(smiles_dict["Name"].values()):
        chg_file = glob.glob(key_dir + "/mol_dft_eng_m06/%s.chg" % name)
        assert len(chg_file) == 1
        chg_file = chg_file[0]
        react_atoms = smiles_dict["React_atom"][id]
        react_atoms = [int(each) for each in react_atoms.split()]
        return_dict = reading_method(chg_file)
        charges = [[] for _ in range(4)]
        for symbol in ['C', 'H', 'O', 'N']:
            charges = return_dict.iloc[react_atoms].loc[return_dict['symbol'] == symbol]['charge']
            if len(charges) != 0:
                smiles_dict['%s_max' % symbol][id] = max(charges)
                smiles_dict['%s_min' % symbol][id] = min(charges)
    pd.DataFrame(smiles_dict).to_csv(key_dir + "/" + return_file, index=False)

def collect_std_Resp_chg(key_dir, return_file="React_atom_resp.csv", reading_method=format_change.read_chg_from_sdf):
    
    smiles_dict = pd.read_csv(key_dir + "/React_atom.csv").to_dict()
    smiles_dict["C_max"], smiles_dict["C_min"],smiles_dict["N_max"], smiles_dict["N_min"] = {}, {}, {}, {}
    smiles_dict["O_max"], smiles_dict["O_min"],smiles_dict["H_max"], smiles_dict["H_min"] = {}, {}, {}, {}
    smiles_dict["charge_select"], smiles_dict["charge_giveup"] = {}, {}
    for id, name in enumerate(smiles_dict["Name"].values()):
        chg_file = glob.glob(key_dir + "/sdf_charge/%s*e78.sdf" % name)
        assert len(chg_file) == 1
        chg_file = chg_file[0]
        react_atoms = smiles_dict["React_atom"][id]
        react_atoms = [int(each) for each in react_atoms.split()]
        return_dict = reading_method(chg_file)
        charges = [[] for _ in range(4)]
        for symbol in ['C', 'H', 'O', 'N']:
            charges = return_dict.iloc[react_atoms].loc[return_dict['symbol'] == symbol]['charge']
            if len(charges) != 0:
                smiles_dict['%s_max' % symbol][id] = max(charges)
                smiles_dict['%s_min' % symbol][id] = min(charges)
    pd.DataFrame(smiles_dict).to_csv(key_dir + "/" + return_file, index=False)

def collect_reaction_file(reaction_file_dir, target_dir, smiles_dir, smiles_csv_dir, sugan=True, acquire_title = True, n_cores = 60, check_charge=True, max_attemp_num=20, acquire_map=False):
    import pickle
    def write_comfirm_dict(comfirm_dict, Reaction_Index, Diene_Index, Ene_Index, Structure_id, conf_id, comfirm_dict_idx, product_G=None, product=None, title=None):
        comfirm_dict['Reaction_Index'][comfirm_dict_idx] = Reaction_Index
        comfirm_dict['Diene_Index'][comfirm_dict_idx] = Diene_Index
        comfirm_dict['Ene_Index'][comfirm_dict_idx] = Ene_Index
        comfirm_dict['Structure_id'][comfirm_dict_idx] = Structure_id
        comfirm_dict['conf_id'][comfirm_dict_idx] = conf_id
        if product_G != None:
            comfirm_dict['product_G'][comfirm_dict_idx] = product_G
        if product != None: comfirm_dict['Product'][comfirm_dict_idx] = product
        if title != None: comfirm_dict['Title'][comfirm_dict_idx] = " ".join([str(num) for num in title])
        comfirm_dict_idx += 1
        return comfirm_dict, comfirm_dict_idx + 1
    SMILES_DIR = smiles_dir
    QM_MAP = SMILES_DIR + "/" + "all_property_detailSterimol_witharea.pkl"
    if acquire_map:
        with open(QM_MAP, "rb") as f:
            target_map = pickle.load(f)
    new_reaction_csv_file = target_dir + '/' + os.path.split(reaction_file_dir)[-1]
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    if reaction_file_dir != new_reaction_csv_file:
        shutil.copy(reaction_file_dir, new_reaction_csv_file)
    reaction_csv_file = pd.read_csv(new_reaction_csv_file)
    smiles_dict = pd.read_csv(smiles_csv_dir)
    comfirm_dict = {"Reaction_Index":{}, "Diene_Index":{}, "Ene_Index":{}, "Product":{}, "Title":{},"Structure_id":{}, "product_G":{}, "product_E":{}, "TS_G":{}, "TS_E":{}, "deltaG":{}, "deltaGa":{}, "deltaE":{}, "deltaEa":{}, "conf_id":{}}
    all_smiles = smiles_dict["Smiles"].to_list()
    smiles_dir = os.path.split(smiles_csv_dir)[0]
    mol_file_dir = target_dir + '/mol/'
    if not os.path.isdir(mol_file_dir):
        os.mkdir(mol_file_dir)
    all_mols = []
    all_mols_name = []
    all_titles = []
    
    comfirm_dict_idx = 0
    for idx, diene in tqdm(enumerate(reaction_csv_file['Diene'])):
        error_id = 0
        diene = reaction_csv_file['Diene'][idx]
        ene = reaction_csv_file['Ene'][idx]
        if acquire_title:
            acquire_title = reaction_csv_file["Title"][idx]
        if not (diene in all_smiles and ene in all_smiles):
            print(diene, ene, "not in All_Smiles")
            raise TypeError
        diene_csv_id = all_smiles.index(diene)
        ene_csv_id = all_smiles.index(ene)
        diene_id = smiles_dict["Index"][diene_csv_id]
        ene_id = smiles_dict["Index"][ene_csv_id]
        if smiles_dict['Stable_conf_id'][diene_csv_id] < 0 or smiles_dict['Stable_conf_id'][ene_csv_id] < 0:
            print(diene, ene, "either one is unstable")
            error_id = -1
            comfirm_dict, comfirm_dict_idx = write_comfirm_dict(comfirm_dict, idx, diene_id, ene_id, 0, error_id, comfirm_dict_idx, product_G="reactant unstable")
            continue
        diene_conf_id = smiles_dict["Stable_conf_id"][diene_csv_id]
        ene_conf_id = smiles_dict["Stable_conf_id"][ene_csv_id]

        diene_mol_dir = smiles_dir + '/mol/smilesid_%.5d.mol' % diene_id
        diene_mol = Chem.MolFromMolFile(diene_mol_dir, removeHs=False)
        diene_log_dir = smiles_dir + "/mol_dft_eng/smilesid_%.5d_%.4d.log" % (diene_id, diene_conf_id)
        diene_log = logfile_process.Logfile(diene_log_dir)
        atom_list = diene_log.symbol_list
        position = diene_log.first_atom_position
        diene_mol = xtb_process.xtb_to_mol(diene_mol, [atom_list], [position], 1)
        diene_mol = cycle_process.add_conformer(diene_mol, max_attemp_num - 1)
        assert len(diene_mol.GetConformers()) !=0

        ene_mol_dir = smiles_dir + '/mol/smilesid_%.5d.mol' % ene_id
        ene_mol = Chem.MolFromMolFile(ene_mol_dir, removeHs=False)
        ene_log_dir = smiles_dir + "/mol_dft_eng/smilesid_%.5d_%.4d.log" % (ene_id, ene_conf_id)
        ene_log = logfile_process.Logfile(ene_log_dir)
        atom_list = ene_log.symbol_list
        position = ene_log.first_atom_position
        ene_mol = xtb_process.xtb_to_mol(ene_mol, [atom_list], [position], 1)
        ene_mol = cycle_process.add_conformer(ene_mol, max_attemp_num - 1)
        assert len(ene_mol.GetConformers()) !=0

        if check_charge:
            banned_diene_title = []
            banned_ene_title = []
            restrict = [[0.14, -0.2], [-0.04, -0.18], [-0.00, -0.00]]
            try:
                diene_chg_file = glob.glob(smiles_dir + "/mol_dft_eng_m06/smilesid_%.5d*.chg" % diene_id)[0]
                diene_return_dict = format_change.read_chg_file(diene_chg_file)
                ene_chg_file = glob.glob(smiles_dir + "/mol_dft_eng_m06/smilesid_%.5d*.chg" % ene_id)[0]
                ene_return_dict = format_change.read_chg_file(ene_chg_file)
                for diene_index in cycle_process.find_all_diene(diene_mol):
                    for symbol_id, symbol in enumerate(['C', 'N', 'O']):
                        charges = diene_return_dict.iloc[diene_index].loc[diene_return_dict['symbol'] == symbol]['charge']
                        if len(charges) != 0:
                            if max(charges) > restrict[symbol_id][0] or min(charges) < restrict[symbol_id][1]:
                                banned_diene_title.append([diene_index[0], diene_index[-1]])
                for ene_index in cycle_process.dieno_atom_Idx(ene_mol):
                    for symbol_id, symbol in enumerate(['C', 'N', 'O']):
                        charges = ene_return_dict.iloc[ene_index].loc[ene_return_dict['symbol'] == symbol]['charge']
                        if len(charges) != 0:
                            if max(charges) > restrict[symbol_id][0] or min(charges) < restrict[symbol_id][1]:
                                banned_ene_title.append([ene_index[0], ene_index[-1]])

            except:
                print(diene, ene, "either one did't calc Hirshfeld charge")
                error_id = -1
                comfirm_dict, comfirm_dict_idx = write_comfirm_dict(comfirm_dict, idx, diene_id, ene_id, 0, error_id, comfirm_dict_idx, product_G="Hirshfeld charge")
                continue

        break_num = 0
        # range_time = 10
        # if check_charge:
        #     range_time = 1
        # for each in range(range_time):
        # try:
        molgroup = cycle_process.react(dieno_mol = ene_mol, diene_mol = diene_mol,select_diene=150, distence=3)
        # except:
        #     molgroup = []
        for structure_id, [eng, mol, title] in enumerate(molgroup):
            if " ".join([str(num) for num in title]) == acquire_title or acquire_title == False:
                if check_charge:
                    if title[:2] in banned_diene_title or title[2:4] in banned_ene_title:
                        comfirm_dict, comfirm_dict_idx = write_comfirm_dict(comfirm_dict, idx, diene_id, ene_id, 0, -1, comfirm_dict_idx, product_G="product unstable")
                        continue
                if acquire_map:
                    diene_atoma, diene_atomb, ene_atomc, ene_atomd, _, title_1 = [int(each) for each in title[:6]]
                    diene_key = "%.5d %d %d %d" % (diene_id, 1, diene_atoma, diene_atomb)
                    ene_key = "%.5d %d %d %d" % (ene_id, 0, ene_atomc, ene_atomd)
                    # if diene_key not in target_map.keys() or ene_key not in target_map.keys():
                    #     continue
                mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
                Chem.MolToMolFile(mol, mol_file_dir + "/%s.mol" % mol_name)
                all_mols.append(mol)
                all_titles.append(title)
                all_mols_name.append(mol_name)
                product = Chem.MolToSmiles(mol)
                comfirm_dict, comfirm_dict_idx = write_comfirm_dict(comfirm_dict, idx, diene_id, ene_id, structure_id, error_id, comfirm_dict_idx, product=product, title=title)
        if len(molgroup) == 0:
            print("Diene %s, Ene %s, not find structure" % (all_smiles[diene_id], all_smiles[ene_id]))
            error_id = -1
            comfirm_dict, comfirm_dict_idx = write_comfirm_dict(comfirm_dict, idx, diene_id, ene_id, 0, error_id, comfirm_dict_idx, product_G="product unstable")
            continue
            
    with open(target_dir + '/' + 'xtb_title.txt', 'wt') as f:
        for mol_name, each in zip(all_mols_name, all_titles):
            f.write(str(mol_name) + "#")
            f.write(" ".join([str(num) for num in each]))
            f.write('\n')
    xtb_process.xtb_main(all_mols_name, all_mols, dir_path=target_dir + '/' + 'first_xtb', que="gamma", core=n_cores)
    # xtb_process.shift_to_parra(target_dir + "/first_xtb")
    # if sugan:
    #     pbs_files = glob.glob(target_dir + '/' + 'first_xtb/*.pbs')
    #     for pbs_file in pbs_files:
    #         with open(pbs_file, "wt",  newline='\n') as f:
    #             number = int(pbs_file.split(".pbs")[0].split("_")[-1])
    #             f.write("#!/bin/bash\n#SBATCH -J g16\n#SBATCH -N 1\n#SBATCH --ntasks-per-node=32\n#SBATCH -p hfacnormal01\n\nroot=`pwd`\nrootdir=charg_0_%d\nfolders=`ls $root/$rootdir/`\n" % number)
    #             f.write("for folder in $folders\ndo\n    cd $root/$rootdir/$folder\n    crest $folder.xyz -T 28 -gfn2 -chrg 0 -uhf 0 -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\ndone\n")
    #     with open(target_dir + '/' + 'first_xtb/suball', "wt",  newline='\n') as f:
    #         for pbs_file in pbs_files:
    #             name = os.path.split(pbs_file)[-1]
    #             f.write("sbatch %s\n" % name)
    a = pd.DataFrame(comfirm_dict)
    a.to_csv(target_dir + '/' + 'Result.csv', index=False)


def reaction_result_analysis(target_dir, opt_name='mol_dft', eng_name='mol_dft_eng', smiles_csv_dir=DIENE_ENE_DIR + "/all_smiles.csv"):
    opt_file_dir = target_dir + "/" + opt_name
    eng_dir = target_dir + "/" + eng_name
    reaction_dict = pd.read_csv(target_dir + "/" + "Result.csv").to_dict()
    smiles_dict = pd.read_csv(smiles_csv_dir).to_dict()
    for idx, _ in tqdm(enumerate(list(reaction_dict["Reaction_Index"]))):
        try:
            diene_id = reaction_dict["Diene_Index"][idx]
            ene_id = reaction_dict['Ene_Index'][idx]
            structure_id = reaction_dict["Structure_id"][idx]
            mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
            mol_file = glob.glob(target_dir + "/mol/%s.mol" % mol_name) 
            if str(reaction_dict['product_G'][idx]) != str(math.nan):
                reaction_dict["conf_id"][idx] = -1
                continue
            if len(mol_file) == 0:
                reaction_dict["product_G"][idx] = "RDKIT Generate Fail"
                reaction_dict["conf_id"][idx] = -1
                continue
            opt_log_files = glob.glob(opt_file_dir + "/" + mol_name + "*.log")
            if len(opt_log_files) == 0:
                reaction_dict["product_G"][idx] = "DFT OPT Fail"
                reaction_dict["conf_id"][idx] = -1
                continue 
            all_engs, all_conf_id, all_E_engs = read_engs(opt_log_files, eng_dir, returnE=1)
            if len(all_engs) == 0:
                reaction_dict["product_G"][idx] = "DFT ENG Fail"
                reaction_dict["conf_id"][idx] = -1
                continue   
            min_conf_idx = np.argmin(all_engs)
            min_engs = all_engs[min_conf_idx]
            min_E_engs = all_E_engs[min_conf_idx]
            min_conf_idx = all_conf_id[min_conf_idx]
            reaction_dict["product_G"][idx] = min_engs
            reaction_dict["conf_id"][idx] = min_conf_idx
            reaction_dict["product_E"][idx] = min_E_engs
            diene_csv_id = list(smiles_dict['Index'].values()).index(diene_id)
            ene_csv_id = list(smiles_dict['Index'].values()).index(ene_id)
            diene_eng = float(smiles_dict['G/Hatree'][diene_csv_id])
            ene_eng = float(smiles_dict['G/Hatree'][ene_csv_id])
            deltag = 627.5 * (min_engs - diene_eng - ene_eng)
            reaction_dict['deltaG'][idx] = deltag

            diene_eng = float(smiles_dict['E/Hatree'][diene_csv_id])
            ene_eng = float(smiles_dict['E/Hatree'][ene_csv_id])
            deltag = 627.5 * (min_E_engs - diene_eng - ene_eng)
            reaction_dict['deltaE'][idx] = deltag
        except:
            continue
    reaction_dict = pd.DataFrame(reaction_dict)
    reaction_dict.to_csv(target_dir + "/" + "Result.csv", index=False)

def reaction_calc_om(target_dir, SELECT_BY_ENG=True, smiles_mol_dir=DIENE_ENE_DIR , opt_name = "mol_dft", om_name = "om", target_cos = -0.9396, Difreeze=True, use_gauss=True):
    opt_file_dir = target_dir + "/" + opt_name
    om_dir = target_dir + "/" + om_name
    if not os.path.isdir(om_dir):
        os.mkdir(om_dir)
    xtb_input = []
    reaction_dict = pd.read_csv(target_dir + "/" + "Result.csv").to_dict()
    for idx, _ in enumerate(list(reaction_dict["Reaction_Index"])):
        conf_id = reaction_dict["conf_id"][idx]
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        structure_id = reaction_dict["Structure_id"][idx]
        deltag = reaction_dict['deltaG'][idx]
        mol_name = "%.5d_%.5d_%.5d_%.4d" % (diene_id, ene_id, structure_id, conf_id)
        diene_mol = glob.glob(smiles_mol_dir + '/mol/smilesid_%.5d*.mol' % diene_id)[0]
        diene_mol = Chem.MolFromMolFile(diene_mol, removeHs=False)
        if conf_id < 0:
            print(mol_name, "Fail in DFT")
            continue
        log_file = glob.glob(opt_file_dir + "/%s.log" % mol_name) 
        if len(log_file) == 0:
            continue
        log_file = log_file[0]
        opt_log = logfile_process.Logfile(log_file)
        
        if not SELECT_BY_ENG or deltag < 10:
            if use_gauss:
                cycle_process.eng_to_om(opt_log,diene_mol, new_dir = om_dir, target_cos=target_cos, difreeze=Difreeze)
            else:
                title, symbol_list, new2_position, charge, diene_atom_lists = cycle_process.eng_to_om(opt_log,diene_mol, new_dir = om_dir, target_cos=target_cos, difreeze=Difreeze, write_gjf=False)
                title = [int(each) for each in title.split()]
                if Difreeze == 1:
                    constrain_atoms = diene_atom_lists + [title[2] + title[4], title[3] + title[4]]
                    constrain_atoms = [each + 1 for each in constrain_atoms]
                    constrain_atoms.sort()
                    dist_const = None
                else:
                    constrain_atoms = None
                    dist_const = [[title[0] + 1, title[2] + title[4] + 1], [title[1] + 1, title[3] + title[4] + 1]]
                xtb_input.append([mol_name, title, symbol_list, new2_position, charge, constrain_atoms, dist_const])
    if not use_gauss:
        xtb_process.xtb_om(xtb_input, dir_path=om_dir, que="gamma", core=70)

    # target_cos = -0.9396926
    # target_cos = -0.7660444
    # target_cos = -0.5

def reaction_calc_om_2( target_dir, 
                        gaussian_input = True, 
                        smiles_mol_dir=DIENE_ENE_DIR , 
                        om_name1 = "om", 
                        om_name2 = "om2", 
                        target_cos = -0.5, 
                        Freeze=True,
                        Difreeze=True, 
                        use_gauss=True,
                        method='opt=modredundant freq b3lyp/6-31g(d) em=gd3bj',
                        ):
    om_dir_1 = target_dir + "/" + om_name1
    om_dir_2 = target_dir + "/" + om_name2
    if not os.path.isdir(om_dir_2):
        os.mkdir(om_dir_2)
    reaction_dict = pd.read_csv(target_dir + "/" + "Result.csv").to_dict()
    # target_cos = -0.9396926
    # target_cos = -0.7660444
    # target_cos = -0.5
    xtb_input = []
    for idx, _ in tqdm(enumerate(list(reaction_dict["Reaction_Index"]))):
        conf_id = reaction_dict["conf_id"][idx]
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        structure_id = reaction_dict["Structure_id"][idx]

        mol_name = "%.5d_%.5d_%.5d_%.4d" % (diene_id, ene_id, structure_id, conf_id)
        # print("Process: %s\r" % mol_name)
        diene_mol = glob.glob(smiles_mol_dir + '/mol/smilesid_%.5d.mol' % diene_id)[-1]
        diene_mol = Chem.MolFromMolFile(diene_mol, removeHs=False) 
        if conf_id < 0:
            print(mol_name, "Fail in DFT")
            continue
        if gaussian_input:
            log_file = glob.glob(om_dir_1 + "/%s.log" % mol_name) 
            if len(log_file) == 0:
                continue
            log_file = log_file[0]
            opt_log = logfile_process.Logfile(log_file)
        else:
            xtb_dir = glob.glob(target_dir + "/" + om_name1 + '/*/' + "%s_0" % mol_name)
            if len(xtb_dir) == 0:
                continue
            xtb_dir = xtb_dir[0]
            charge = int(xtb_dir.split("\\")[-2].split("_")[-2])
            xtb_file = xtb_dir + "/crest_conformers.xyz"
            try:
                atom_lists, positions = xtb_process.read_xyz(xtb_file)
            except:
                xtb_file = xtb_dir + f"/{mol_name}_0.xyz"
                atom_lists, positions = xtb_process.read_xyz(xtb_file) 
            atom_list, position = atom_lists[0], positions[0]
            title = reaction_dict["Title"][idx]
            title = [int(each) for each in title.split()]
            opt_log = "%s.gjf" % mol_name, title, charge, atom_list, position
        if use_gauss:
            cycle_process.om_to_om2(opt_log,diene_mol, om_dir_2, target_cos=target_cos, freeze=Freeze,difreeze=Difreeze, write_gjf=True, method=method)
        else:
            title, symbol_list, new2_position, charge, diene_atom_lists = cycle_process.om_to_om2(opt_log,diene_mol, om_dir_2, target_cos=target_cos, difreeze=Difreeze, write_gjf=False)
            title = [int(each) for each in title.split()]
            if Difreeze:
                constrain_atoms = diene_atom_lists[0] + [title[2] + title[4], title[3] + title[4]]
                constrain_atoms = [each + 1 for each in constrain_atoms]
                constrain_atoms.sort()
                dist_const = None
            else:
                constrain_atoms = None
                dist_const = [[title[0], title[2] + title[4]], [title[1], title[3] + title[4]]]

            xtb_input.append([mol_name, title, symbol_list, new2_position, charge, constrain_atoms, dist_const])
    if not use_gauss:
        xtb_process.xtb_om(xtb_input, dir_path=om_dir_2, que="gamma", core=70)


def reaction_calc_ts(target_dir, om_name="om", ts_name="ts"):
    om_file_dir = target_dir + "/" + om_name
    ts_dir = target_dir + "/" + ts_name
    if not os.path.isdir(ts_dir):
        os.mkdir(ts_dir)
    om_log_files = glob.glob(om_file_dir + "/*.log")
    for om_log_file in om_log_files:
        om_log = logfile_process.Logfile(om_log_file)
        assert om_log.bond_attach
        cycle_process.om_to_ts(om_log, 1, new_dir=ts_dir)

def reaction_calc_irc(target_dir, ts_name='ts', irc_name='irc', methods = None):
    # 仅针对频率较低或者振动方向错误的
    ts_file_dir = target_dir + "/" + ts_name
    irc_dir = target_dir + "/" + irc_name
    if not os.path.isdir(irc_dir):
        os.mkdir(irc_dir)
    ts_log_files = glob.glob(ts_file_dir + "/*.log")
    for ts_log_file in ts_log_files:
        ts_log = logfile_process.Logfile(ts_log_file)
        assert ts_log.bond_attach
        if ts_log.is_right_ts:
            if float(ts_log.first_unreal_freq) <= -300:
                continue
            else:
                print("%s May have wrong vibration freq: %.4f!!! " % (ts_log.file_dir, float(ts_log.first_unreal_freq)))
        else:
            print("%s May have wrong vibration direction!!! " % ts_log.file_dir)
        cycle_process.ts_to_irc(ts_log, new_dir=irc_dir, methods = methods)


def ts_SPE_DFT_calc(target_dir, ts_name='ts', eng_name = 'ts_eng', method="b3lyp/6-311+g(d,p) em=gd3bj"):
    ts_file_dir = target_dir + "/" + ts_name
    eng_dir = target_dir + "/" + eng_name
    mol_files = glob.glob(target_dir + "/mol/*.mol")
    log_files = glob.glob(ts_file_dir + "/" + "*.log")
    for log_file in tqdm(log_files):
        new_log_name = eng_dir + "/" + os.path.split(log_file)[-1].split('.')[0] + ".gjf" 
        opt_log = logfile_process.Logfile(log_file)
        assert len(opt_log.running_positions) != 0
        title, charge, symbol_list, position,= opt_log.title, opt_log.charge, opt_log.symbol_list, opt_log.running_positions[-1]
        title = " ".join(str(each) for each in title)
        format_change.block_to_gjf(symbol_list, position, new_log_name, charge, title,
                    method=method)


def reaction_irc_select(target_dir, ts_name='ts', irc_name='irc', dust_bin_name='ts_irc_fail'):
    ts_file_dir = target_dir + "/" + ts_name
    irc_dir = target_dir + "/" + irc_name
    dust_bin_dir = target_dir + "/" + dust_bin_name
    ts_log_files = glob.glob(ts_file_dir + "/*.log")
    for ts_log_file in tqdm(ts_log_files):
        ts_log = logfile_process.Logfile(ts_log_file)
        assert ts_log.bond_attach
        check_irc = 0
        if ts_log.is_right_ts:
            if float(ts_log.first_unreal_freq) <= -300:
                continue
            else:
                check_irc = 1
        else:
            check_irc = 1
        if check_irc:
            irc_files = glob.glob(irc_dir + "/%s*.log" % os.path.split(ts_log.file_dir)[-1].split(".")[0])
            if len(irc_files) < 1:
                print(ts_log.file_dir, "did't find right irc files")
                new_log_name = dust_bin_dir + "/" + os.path.split(ts_log_file)[-1] 
                new_log_name =  new_log_name.split(".")[0] + "%s.log" % ts_log.file_type
                if not os.path.isdir(dust_bin_dir):
                    os.mkdir(dust_bin_dir) 
                shutil.move(ts_log_file, new_log_name)

def ts_up_down_check(target_dir, ts_name='ts', dust_bin="up_down_fail_ts"):
    ts_file_dir = target_dir + "/" + ts_name
    dust_bin_dir = target_dir + "/" + dust_bin
    smiles_mol_dir=DIENE_ENE_DIR 
    if not os.path.isdir(dust_bin_dir):
        os.mkdir(dust_bin_dir)
    reaction_dict = pd.read_csv(target_dir + "/" + "Result.csv").to_dict()
    error_num = 0
    for idx, _ in enumerate(list(reaction_dict["Reaction_Index"])):
        conf_id = reaction_dict["conf_id"][idx]
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        structure_id = reaction_dict["Structure_id"][idx]
        conf_id = reaction_dict["conf_id"][idx]
        title = reaction_dict["Title"][idx]
        if str(title) == "nan":
            continue
        diene_num1, diene_num2, dieno_num1, dieno_num2, start = [int(each) for each in title.split()][:5]
        mol_name = "%.5d_%.5d_%.5d*" % (diene_id, ene_id, structure_id)
        diene_mol = glob.glob(smiles_mol_dir + '/mol/smilesid_%.5d.mol' % diene_id)[-1]
        diene_mol = Chem.MolFromMolFile(diene_mol, removeHs=False)
        
        log_files = glob.glob(ts_file_dir + "/%s.log" % mol_name) 
        diene_atom_lists = cycle_process.diene_atom_Idx(diene_mol, select_diene=0)
        for each in diene_atom_lists:
            if each[0] == diene_num1 and each[-1] == diene_num2:
                atomb_id, atomc_id = each[1:3]
                break
        for log_file in log_files:
            log = logfile_process.Logfile(log_file)
            position = log.running_positions[-1]
            diene_point1 = position[diene_num1][:3]
            diene_point2 = position[diene_num2][:3]
            dieno_point1 = position[dieno_num1 + start][:3]
            dieno_point2 = position[dieno_num2 + start][:3]
            diene_point3 = position[atomb_id][:3]
            new_position = cycle_process.trfm_rot(diene_point1, diene_point2, diene_point3, position)
            dieno_point2 = new_position[dieno_num2 + start][:3]
            if dieno_point2[2] < 0: 
                print(log.file_dir)
                error_num += 1
                new_file = dust_bin_dir + "/%s" % os.path.split(log_file)[-1]
                shutil.move(log_file, new_file)
            


def add_product_smiles(target_dir):
    target_dict = pd.read_csv(target_dir + "/Result.csv").to_dict()
    target_dict["Product"] = {}
    mol_dir = target_dir + "/mol"
    for idx in range(len(target_dict['Diene_Index'])):
        diene_id = target_dict["Diene_Index"][idx]
        ene_id = target_dict["Ene_Index"][idx]
        structure_id = target_dict['Structure_id'][idx]
        mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
        mol_files = glob.glob(mol_dir + "/%s.mol" % mol_name)
        if len(mol_files) != 1:
            smiles = ""
        else:
            mol_file = mol_files[0]
            mol = Chem.MolFromMolFile(mol_file)
            smiles = Chem.MolToSmiles(mol)
            Tool.stablize_smileses([smiles])
        target_dict["Product"][idx] = smiles
    target_dict = pd.DataFrame(target_dict)
    target_dict.to_csv(target_dir + "/Result.csv", index=0)

def check_product_stereo(target_dir):
    target_dict = pd.read_csv(target_dir + "/Result.csv").to_dict()
    target_dict["Product"] = {}
    mol_dir = DIENE_ENE_DIR +'/mol'
    new_ts_dir = target_dir + "/ts_wrong_stereo/"
    if not os.path.isdir(new_ts_dir): os.mkdir(new_ts_dir)
    for idx in tqdm(range(len(target_dict['Diene_Index']))):
        diene_id = target_dict["Diene_Index"][idx]
        ene_id = target_dict["Ene_Index"][idx]
        structure_id = target_dict['Structure_id'][idx]
        try:
            title = target_dict["Title"][idx]
            title = [int(each) for each in title.split()]
            split_index = title[4]
            mol_str = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
            diene_mol_file = glob.glob(mol_dir + '/smilesid_%.5d*' % diene_id)[0]
            ene_mol_file = glob.glob(mol_dir + '/smilesid_%.5d*' % ene_id)[0]
            log_files = glob.glob(target_dir + '/ts/%s*.log' % mol_str)
        except:
            continue
        if len(log_files) == 0:
            continue
        for log_file in log_files:
            opt_log = logfile_process.Logfile(log_file)
            atom_list = opt_log.symbol_list
            position = opt_log.running_positions[-1]
            diene_is_right = cycle_process.check_stereo(diene_mol_file,atom_list[:split_index], position[:split_index])
            ene_is_right = cycle_process.check_stereo(ene_mol_file,atom_list[split_index:], position[split_index:])
            if not diene_is_right or not ene_is_right:
                print(diene_id, ene_id, "has wrong stereo!")
                new_log_file = new_ts_dir + os.path.split(log_file)[-1]
                shutil.move(log_file, new_log_file)

def get_special_properties(target_csv, smiles_dir=DIENE_ENE_DIR):
    reaction_dict = pd.read_csv(target_csv)
    dieneas, eneas = [],[]
    smiles_dict = pd.read_csv(smiles_dir + "/all_smiles.csv")
    for idx, row in tqdm(reaction_dict.iterrows()):
        diene_id = row["Diene_Index"]
        ene_id = row['Ene_Index']
        title = row['Title']
        atoma_id, atomd_id, atome_id, atomf_id = [int(each) for each in title.split()[:4]]
        diene_csv_id = list(smiles_dict['Index'].values).index(diene_id)
        ene_csv_id = list(smiles_dict['Index'].values).index(ene_id)
        diene_conf_id = smiles_dict["Stable_conf_id"][diene_csv_id]
        ene_conf_id = smiles_dict["Stable_conf_id"][ene_csv_id]
        diene_mol_dir = smiles_dir + '/mol/smilesid_%.5d.mol' % diene_id
        diene_mol = Chem.MolFromMolFile(diene_mol_dir, removeHs=False)
        diene_log_dir = smiles_dir + "/mol_dft/smilesid_%.5d_%.4d.log" % (diene_id, diene_conf_id)
        diene_log = logfile_process.Logfile(diene_log_dir)
        ene_mol_dir = smiles_dir + '/mol/smilesid_%.5d.mol' % ene_id
        ene_mol = Chem.MolFromMolFile(ene_mol_dir, removeHs=False)
        ene_log_dir = smiles_dir + "/mol_dft/smilesid_%.5d_%.4d.log" % (ene_id, ene_conf_id)
        ene_log = logfile_process.Logfile(ene_log_dir)

        diene_atom_lists = cycle_process.diene_atom_Idx(diene_mol, select_diene=0)
        for each in diene_atom_lists:
            if each[0] == atoma_id and each[-1] == atomd_id:
                atomb_id, atomc_id = each[1:3]
                break
        diene_position = diene_log.running_positions[-1]
        ene_position = ene_log.running_positions[-1]
        a_neighbor = [each.GetIdx() for each in diene_mol.GetAtomWithIdx(atoma_id).GetNeighbors() if each.GetIdx() != atomb_id]
        d_neighbor = [each.GetIdx() for each in diene_mol.GetAtomWithIdx(atomd_id).GetNeighbors() if each.GetIdx() != atomc_id]
        e_neighbor = [each.GetIdx() for each in ene_mol.GetAtomWithIdx(atome_id).GetNeighbors() if each.GetIdx() != atomf_id]
        f_neighbor = [each.GetIdx() for each in ene_mol.GetAtomWithIdx(atomf_id).GetNeighbors() if each.GetIdx() != atome_id]
        diene_angles = [Tool.get_bond_angle_deg(diene_position[each], diene_position[atoma_id], diene_position[atomb_id]) for each in a_neighbor] + \
        [Tool.get_bond_angle_deg(diene_position[atomc_id], diene_position[atomd_id], diene_position[each]) for each in d_neighbor]
        ene_angles = [Tool.get_bond_angle_deg(ene_position[each], ene_position[atome_id], ene_position[atomf_id]) for each in e_neighbor] + \
        [Tool.get_bond_angle_deg(ene_position[atome_id], ene_position[atomf_id], ene_position[each]) for each in f_neighbor]
        if len(a_neighbor) + len(d_neighbor) == 0:diene_angles = [-0.5, -0.5]
        if len(e_neighbor) + len(f_neighbor) == 0:ene_angles = [-0.5, -0.5]
        diene_angles = [np.abs(np.arccos(each) * 180 / np.pi - 120) for each in diene_angles]
        ene_angles = [np.abs(np.arccos(each) * 180 / np.pi - 120) for each in ene_angles]
        dieneas.append([max(diene_angles), min(diene_angles)])
        eneas.append([max(ene_angles), min(ene_angles)])
    return dieneas, eneas


def reaction_ts_result_analysis(target_dir, smiles_csv_dir=DIENE_ENE_DIR + "/all_smiles.csv", write_title=False, om_name='om', ts_name='ts', ts_eng_name='ts_eng', sep_react_path = "sep_react"):
    ts_file_dir = target_dir + "/" + ts_name
    eng_dir = target_dir + "/" + ts_eng_name
    eng_solvent_dir = target_dir + "/" + 'ts_eng_solvent'
    title_dict = read_title(target_dir)
    reaction_dict = pd.read_csv(target_dir + "/" + "Result.csv").to_dict()
    smiles_dict = pd.read_csv(smiles_csv_dir).to_dict()
    smiles_idxs = list(smiles_dict["Index"].values())
    reaction_dict["Diene"] = {}
    reaction_dict["Ene"] = {}
    reaction_dict["Diene_Distort"] = {}
    reaction_dict["Ene_Distort"] = {}
    reaction_dict["Interaction"] = {}
    reaction_dict["Special_Diene_E"] = {}
    reaction_dict["Special_Diene_G"] = {}
    reaction_dict["TS_G(Solvent)"] = {}
    reaction_dict["deltaGa(Solvent)"] = {}
    for idx, _ in tqdm(enumerate(list(reaction_dict["Reaction_Index"]))):
        # try:
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        diene_csv_id = smiles_idxs.index(diene_id)
        ene_csv_id = smiles_idxs.index(ene_id)
        reaction_dict["Diene"][idx] = smiles_dict["Smiles"][diene_csv_id]
        reaction_dict["Ene"][idx] = smiles_dict["Smiles"][ene_csv_id]
        structure_id = reaction_dict["Structure_id"][idx]
        mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
        ts_log_files = glob.glob(ts_file_dir + "/" + mol_name + "*.log")
        if len(ts_log_files) == 0:
            reaction_dict["TS_G"][idx] = "TS OPT Fail"
            reaction_dict["conf_id"][idx] = -1
            continue 
        all_engs, all_conf_id, all_E_engs = read_engs(ts_log_files, eng_dir, returnE=1)
        if len(all_engs) == 0:
            reaction_dict["TS_G"][idx] = "TS ENG Fail"
            reaction_dict["conf_id"][idx] = -1
            continue  
        all_solvent_engs, _, _ = read_engs(ts_log_files, eng_solvent_dir, returnE=1)
        if len(all_solvent_engs) == 0:
            reaction_dict["TS_G(Solvent)"][idx] = "TS ENG Fail"
            reaction_dict["conf_id"][idx] = -1
            continue  
        if write_title:
            reaction_dict["Title"][idx] = title_dict[mol_name] 
        min_conf_idx = np.argmin(all_solvent_engs)
        min_engs = all_engs[min_conf_idx]
        min_solvent_engs = all_solvent_engs[min_conf_idx]
        min_E_engs = all_E_engs[min_conf_idx]
        min_conf_idx = all_conf_id[min_conf_idx]
        reaction_dict["TS_G"][idx] = min_engs
        reaction_dict["TS_G(Solvent)"][idx] = min_solvent_engs
        reaction_dict["TS_E"][idx] = min_E_engs
        reaction_dict["conf_id"][idx] = min_conf_idx
        diene_G = float(smiles_dict['G/Hatree'][diene_csv_id])
        ene_G = float(smiles_dict['G/Hatree'][ene_csv_id])
        diene_G_solvent = float(smiles_dict['G(Solvent)/Hatree'][diene_csv_id])
        ene_G_solvent = float(smiles_dict['G(Solvent)/Hatree'][ene_csv_id])
        diene_E = float(smiles_dict['E/Hatree'][diene_csv_id])
        ene_E = float(smiles_dict['E/Hatree'][ene_csv_id])
        special_diene_files = glob.glob(os.path.join(target_dir, "special_reactant", f"{mol_name}_{min_conf_idx:04}_diene.log"))
        if len(special_diene_files) > 0:
            all_diene_engs, _, all_E_engs = read_engs(special_diene_files, os.path.join(target_dir, "special_reactant_eng"), returnE=1)
            if len(all_diene_engs) > 0:
                diene_G, diene_E = all_diene_engs[0], all_E_engs[0]
                reaction_dict["Special_Diene_E"][idx] = diene_E
                reaction_dict["Special_Diene_G"][idx] = diene_G
        deltaGa = 627.5 * (min_engs - diene_G - ene_G)
        reaction_dict['deltaGa'][idx] = deltaGa
        deltaGa_solvent = 627.5 * (min_solvent_engs - diene_G_solvent - ene_G_solvent)
        reaction_dict['deltaGa(Solvent)'][idx] = deltaGa_solvent
        deltaEa = 627.5 * (min_E_engs - diene_E - ene_E)
        reaction_dict["deltaEa"][idx] = deltaEa
        product_G = float(reaction_dict['product_G'][idx])
        product_E = float(reaction_dict['product_E'][idx])
        deltaG = 627.5 * (product_G - diene_G - ene_G)
        reaction_dict['deltaG'][idx] = deltaG
        deltaE = 627.5 * (product_E - diene_E - ene_E)
        reaction_dict["deltaE"][idx] = deltaE

        if sep_react_path != None:
            diene_log_file = glob.glob(target_dir + "/" + sep_react_path + "/%.5d_%.5d_%.5d_%.4d_diene*.log" % (diene_id, ene_id, structure_id, min_conf_idx))
            if len(diene_log_file) != 1:
                if deltaGa > 0 : print(diene_id, ene_id, structure_id, "check the number of sep_react files: %d" % len(diene_log_file))
                continue
            diene_log_file = diene_log_file[0]
            diene_log = logfile_process.Logfile(diene_log_file)
            diene_ts_eng = diene_log.all_engs[0]
            diene_distort_eng = 627.5 * (diene_ts_eng - diene_E)
            reaction_dict["Diene_Distort"][idx] = diene_distort_eng
            ene_log_file = glob.glob(target_dir + "/" + sep_react_path + "/%.5d_%.5d_%.5d_%.4d_ene*.log" % (diene_id, ene_id, structure_id, min_conf_idx))
            if len(ene_log_file) != 1:
                if deltaGa > 0 : print(diene_id, ene_id, structure_id, "check the number of sep_react files: %d" % len(ene_log_file))
                continue
            ene_log_file = ene_log_file[0]
            ene_log = logfile_process.Logfile(ene_log_file)
            ene_ts_eng = ene_log.all_engs[0]
            ene_distort_eng = 627.5 * (ene_ts_eng - ene_E)
            reaction_dict["Ene_Distort"][idx] = ene_distort_eng
            interaction = diene_distort_eng + ene_distort_eng - deltaEa
            reaction_dict["Interaction"][idx] = interaction
        # except:
        #     continue
    reaction_dict = pd.DataFrame(reaction_dict)
    # return reaction_dict
    reaction_dict.to_csv(target_dir + "/" + "Result_.csv", index=False)

def reaction_analysis(target_dir, opt_name='ts', eng_name='ts_eng', result_csv = 'test.csv'):
    opt_dir = target_dir + "/" + opt_name
    eng_dir = target_dir + "/" + eng_name
    reaction_dict = {"Name":{}, "E(Hatree)":{}, "G(Hatree)":{}}

    opt_logs = glob.glob(opt_dir + "/*.log")
    for idx, log_file in tqdm(enumerate(opt_logs)):
        all_engs, all_conf_id, all_E_engs = read_engs([log_file], eng_dir, returnE=1)
        if len(all_engs) == 0:
            reaction_dict["E(Hatree)"][idx] = "TS ENG Fail"
            reaction_dict["E(Hatree)"][idx] = "TS ENG Fail"
            continue  
        min_conf_idx = np.argmin(all_engs)
        min_engs = all_engs[min_conf_idx]
        min_E_engs = all_E_engs[min_conf_idx]
        min_conf_idx = all_conf_id[min_conf_idx]
        reaction_dict["Name"][idx] = log_file.split("/")[-1]
        reaction_dict["E(Hatree)"][idx] = min_E_engs
        reaction_dict["G(Hatree)"][idx] = min_engs
    reaction_dict = pd.DataFrame(reaction_dict)
    reaction_dict.to_csv(target_dir + "/" + result_csv, index=False)


def reaction_sum_data(target_csv_dir, pred_dict = None,smiles_csv_dir=DIENE_ENE_DIR + "/all_smiles.csv"):
    if pred_dict == None:
        result_dict = {"Reaction_Index":{}, "Diene":{}, "Diene_Index":{}, "Ene":{}, "Ene_Index":{},"Product":{}, "Structure_Id":{},"Title":{}, "deltaG":{}, "deltaGa":{},"deltaE":{}, "deltaEa":{}, "Diene_Distort":{}, "Ene_Distort":{}, "Interaction":{}, "Fail reason":{}}
    else:
        result_dict = pd.read_csv(pred_dict).to_dict()
        result_dict["deltaG"] = {}
        result_dict["deltaGa"] = {}
    reaction_dict = pd.read_csv(target_csv_dir)
    smiles_dict = pd.read_csv(smiles_csv_dir)
    # simple_result_idx = 0
    for idx, _ in enumerate(list(reaction_dict["Reaction_Index"])):
        result_dict["Reaction_Index"] = reaction_dict["Reaction_Index"]
        result_dict["Product"] = reaction_dict["Product"]
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        title = reaction_dict["Title"][idx]
        diene_csv_id = list(smiles_dict['Index'].values).index(diene_id)
        ene_csv_id = list(smiles_dict['Index'].values).index(ene_id)
        diene_smiles = smiles_dict["Smiles"][diene_csv_id]
        ene_smiles = smiles_dict["Smiles"][ene_csv_id]

        result_dict["Diene"][idx] = diene_smiles
        result_dict["Ene"][idx] = ene_smiles
        result_dict["Diene_Index"][idx] = diene_id
        result_dict["Ene_Index"][idx] = ene_id
        result_dict["Title"][idx] = title
        # result_dict["Structure_Id"][idx] = reaction_dict["Structure_id"][idx]
        
        # simple_dict_write = 0
        # if simple_result_idx == 0:
        #     simple_dict_write = 1
        # elif simple_result_dict["Diene"][idx - 1] != diene_smiles and simple_result_dict["Ene"][idx - 1] != ene_smiles:
        #     simple_dict_write = 1
        # if simple_dict_write:
        #     simple_result_dict["Diene"][idx] = diene_smiles
        #     simple_result_dict["Ene"][idx] = ene_smiles
        #     simple_result_dict["Title"][idx] = title

        conf_id = reaction_dict['conf_id'][idx]
        if int(conf_id) == -1:
            if str(reaction_dict['deltaG'][idx]) == str(math.nan):
                result_dict["deltaG"][idx] = -1
                result_dict["deltaGa"][idx] = -1
                # result_dict["Fail reason"][idx] = reaction_dict["product_G"][idx]
            else:
                result_dict["deltaG"][idx] = reaction_dict['deltaG'][idx]
                result_dict["deltaGa"][idx] = -1
                # error_reason = reaction_dict["TS_G"][idx]
                # if float(reaction_dict['deltaG'][idx]) > 0:
                #     error_reason = "Product unstable"
                # result_dict["Fail reason"][idx] = error_reason
            
        else:
            result_dict["deltaG"][idx] = reaction_dict['deltaG'][idx]
            result_dict["deltaGa"][idx] = reaction_dict['deltaGa'][idx]
            result_dict["deltaE"][idx] = reaction_dict['deltaE'][idx]
            result_dict["deltaEa"][idx] = reaction_dict['deltaEa'][idx]
            result_dict["Structure_Id"][idx] = reaction_dict['Structure_id'][idx]
            result_dict["Diene_Distort"][idx] = reaction_dict['Diene_Distort'][idx]
            result_dict["Ene_Distort"][idx] = reaction_dict['Ene_Distort'][idx]
            result_dict["Interaction"][idx] = reaction_dict['Interaction'][idx]
    result_dict = pd.DataFrame(result_dict)
    return result_dict

def ts_Z_E_check(target_dir, smiles_csv_dir=DIENE_ENE_DIR + "/all_smiles.csv", ts_name='ts', move_file='wrong_ZE', ignore_H=True):
    reaction_dict = pd.read_csv(target_dir + '/Result.csv')
    smiles_dict = pd.read_csv(smiles_csv_dir)
    smiles_dir = os.path.split(smiles_csv_dir)[0]
    ts_dir = target_dir + "/" + ts_name
    # mol_dft_dir = target_dir + "/mol_dft"
    if move_file != None:
        wrong_ZE_dir = target_dir + "/" + move_file
        if not os.path.isdir(wrong_ZE_dir):
            os.mkdir(wrong_ZE_dir)
    for idx, _ in enumerate(reaction_dict['Diene_Index']):
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        structure_id = reaction_dict["Structure_id"][idx]
        conf_id = reaction_dict["conf_id"][idx]
        if conf_id == -1:
            continue
        title = reaction_dict["Title"][idx]
        if str(title) == 'nan':
            continue
        titles = [int(each) for each in title.split()]
        mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
        ts_log_files = glob.glob(ts_dir + "/" + mol_name + "*.log")
        if len(ts_log_files) == 0:
            continue
        diene_csv_id = list(smiles_dict['Index'].values).index(diene_id)
        ene_csv_id = list(smiles_dict['Index'].values).index(ene_id)
        diene_conf_id = smiles_dict["Stable_conf_id"][diene_csv_id]
        ene_conf_id = smiles_dict["Stable_conf_id"][ene_csv_id]
        diene_mol_dir = smiles_dir + '/mol/smilesid_%.5d.mol' % diene_id
        diene_mol = Chem.MolFromMolFile(diene_mol_dir, removeHs=False)
        diene_log_dir = smiles_dir + "/mol_dft_eng/smilesid_%.5d_%.4d.log" % (diene_id, diene_conf_id)
        diene_log = logfile_process.Logfile(diene_log_dir)
        diene_position = diene_log.first_atom_position

        ene_mol_dir = smiles_dir + '/mol/smilesid_%.5d.mol' % ene_id
        ene_mol = Chem.MolFromMolFile(ene_mol_dir, removeHs=False)
        ene_log_dir = smiles_dir + "/mol_dft_eng/smilesid_%.5d_%.4d.log" % (ene_id, ene_conf_id)
        ene_log = logfile_process.Logfile(ene_log_dir)
        ene_position = ene_log.first_atom_position

        for ts_log_file in ts_log_files:
            try:
                print("process : %s" % ts_log_file, end='\r')
                is_error = []
                ts_log = logfile_process.Logfile(ts_log_file)
                position_list = ts_log.running_positions[-1]
                cut_num = titles[4]
                ts_diene_position = position_list[:cut_num]
                ts_ene_position = position_list[cut_num:]
                assert len(ts_diene_position) == len(diene_position) and len(ts_ene_position) == len(ene_position)
                for mol, old_position, new_position in zip([diene_mol, ene_mol], [diene_position, ene_position], [ts_diene_position, ts_ene_position]):
                    if len(is_error) > 0:break
                    is_error = cycle_process.check_double_bond_ZE(mol, old_position, new_position, ignore_H=ignore_H)
                if len(is_error) > 0:
                    print(ts_log.file_dir, "TS ZE is wrong!!!!!", is_error)
                    if move_file:
                        old_mol_dft_files = glob.glob(ts_dir + "/%s*.log" % mol_name)
                        for each in old_mol_dft_files:
                            new_file = wrong_ZE_dir + "/%s" % os.path.split(each)[-1]
                            shutil.move(each, new_file)
            except Exception as e:
                print(ts_log_file, e)

def check_ts_stereo(target_dir, ts_name='ts', wrong_ts_name='ts_wrong_stereo'):
    target_dict = pd.read_csv(target_dir + "/Result.csv").to_dict()
    mol_dir = target_dir + '/mol/'
    new_ts_dir = target_dir + "/%s/" % wrong_ts_name
    if not os.path.isdir(new_ts_dir): os.mkdir(new_ts_dir)
    for idx in tqdm(range(len(target_dict['Diene_Index']))):
        diene_id = target_dict["Diene_Index"][idx]
        ene_id = target_dict["Ene_Index"][idx]
        structure_id = target_dict['Structure_id'][idx]
        try:
            mol_str = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
            mol_file = glob.glob(mol_dir + "/%s*.mol" % mol_str)
            log_files = glob.glob(target_dir + '/ts/%s*.log' % mol_str)
        except:
            continue
        if len(log_files) == 0:
            continue
        for log_file in log_files:
            opt_log = logfile_process.Logfile(log_file)
            atom_list = opt_log.symbol_list
            position = opt_log.running_positions[-1]
            is_right = cycle_process.check_stereo(mol_file, atom_list, position)
            if not diene_is_right or not ene_is_right:
                print(diene_id, ene_id, "has wrong stereo!")
                new_log_file = new_ts_dir + os.path.split(log_file)[-1]
                shutil.move(log_file, new_log_file)

def reaction_calc_seperate_reactants(target_dir, ts_name='ts', react_name='sep_react', type='both'):
    ts_file_dir = target_dir + "/" + ts_name
    seperate_reactant_dir = target_dir + "/" + react_name
    if not os.path.isdir(seperate_reactant_dir):
        os.mkdir(seperate_reactant_dir)
    om_log_files = glob.glob(ts_file_dir + "/*.log")
    for om_log_file in om_log_files:
        om_log = logfile_process.Logfile(om_log_file)
        start = om_log.title[4]
        all_symbol_list = om_log.symbol_list
        all_position = om_log.running_positions[-1]
        if type == 'both' or type == 'diene':
            diene_symbol_list = all_symbol_list[:start]
            diene_position = all_position[:start]
            diene_file = seperate_reactant_dir + "/" + os.path.split(om_log_file)[-1].split(".")[0] + "_diene.gjf"
            format_change.block_to_gjf(symbol_list=diene_symbol_list, positions=diene_position, 
                                       file = diene_file, method='b3lyp/6-311+g(d,p) em=gd3bj')
        if type == 'both' or type == 'ene':
            ene_symbol_list = all_symbol_list[start:]
            ene_position = all_position[start:]
            ene_file = seperate_reactant_dir + "/" + os.path.split(om_log_file)[-1].split(".")[0] + "_ene.gjf"
            format_change.block_to_gjf(symbol_list=ene_symbol_list, positions=ene_position, 
                                       file = ene_file, method='b3lyp/6-311+g(d,p) em=gd3bj')

def reaction_distort_interaction_analysis(target_dir, smiles_csv_dir=DIENE_ENE_DIR + "/all_smiles.csv", react_dir='sep_react', type="both"):
    reaction_dict = pd.read_csv(target_dir + "/" + "Result.csv").to_dict()
    smiles_dict = pd.read_csv(smiles_csv_dir).to_dict()
    smiles_idxs = list(smiles_dict["Index"].values())
    reaction_dict["Diene_Distort"] = {}
    reaction_dict["Ene_Distort"] = {}
    reaction_dict["Interaction"] = {}
    for idx, _ in tqdm(enumerate(list(reaction_dict["Reaction_Index"]))):
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        structure_id = reaction_dict['Structure_id'][idx]
        conf_id = reaction_dict['conf_id'][idx]
        deltaGa = reaction_dict['deltaGa'][idx]
        deltaEa = reaction_dict['deltaEa'][idx]
        if type == 'both' or type == 'diene':
            diene_log_file = glob.glob(target_dir + "/" + react_dir + "/%.5d_%.5d_%.5d_%.4d_diene*.log" % (diene_id, ene_id, structure_id, conf_id))
            if len(diene_log_file) != 1:
                if deltaGa > 0 : print(diene_id, ene_id, structure_id, conf_id, "check the number of sep_react files: %d" % len(diene_log_file))
                continue
            diene_log_file = diene_log_file[0]
            diene_log = logfile_process.Logfile(diene_log_file)
            diene_ts_eng = diene_log.all_engs[0]
            diene_mol_eng = smiles_dict['E/Hatree'][smiles_idxs.index(diene_id)]
            diene_distort_eng = 627.5 * (diene_ts_eng - diene_mol_eng)
            reaction_dict["Diene_Distort"][idx] = diene_distort_eng

        if type == 'both' or type == 'ene':
            ene_log_file = glob.glob(target_dir + "/" + react_dir + "/%.5d_%.5d_%.5d_%.4d_ene*.log" % (diene_id, ene_id, structure_id, conf_id))
            if len(ene_log_file) != 1:
                if deltaGa > 0 : print(diene_id, ene_id, structure_id, conf_id, "check the number of sep_react files: %d" % len(ene_log_file))
                continue
            ene_log_file = ene_log_file[0]
            ene_log = logfile_process.Logfile(ene_log_file)
            ene_ts_eng = ene_log.all_engs[0]
            ene_mol_eng = smiles_dict['E/Hatree'][smiles_idxs.index(ene_id)]
            ene_distort_eng = 627.5 * (ene_ts_eng - ene_mol_eng)
            reaction_dict["Ene_Distort"][idx] = ene_distort_eng
        
        if type == "both" and deltaGa > 0:
            interaction = diene_distort_eng + ene_distort_eng - deltaEa
            reaction_dict["Interaction"][idx] = interaction
    return pd.DataFrame(reaction_dict)

def energy_selection(target_dir, eng_name='ts_eng_2', new_eng_name='ts_eng_3'):
    eng_dir = target_dir + "/" + eng_name
    new_eng_dir = target_dir + "/" + new_eng_name
    if not os.path.isdir(new_eng_dir):
        os.mkdir(new_eng_dir)
    reaction_dict = pd.read_csv(target_dir + "/" + "Result.csv").to_dict()
    for idx, _ in enumerate(list(reaction_dict["Reaction_Index"])):
        diene_id = reaction_dict["Diene_Index"][idx]
        ene_id = reaction_dict['Ene_Index'][idx]
        structure_id = reaction_dict["Structure_id"][idx]
        mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
        ts_eng_files = glob.glob(eng_dir + "/" + mol_name + "*.log")
        if len(ts_eng_files) == 0:
            continue
        all_engs = []
        for ts_eng in ts_eng_files:
            eng_log = logfile_process.Logfile(ts_eng)
            all_engs.append(eng_log.all_engs[0])
        min_idx = np.argmin(all_engs)
        old_file = ts_eng_files[min_idx]
        new_file = new_eng_dir + "/" + os.path.split(old_file)[-1]
        shutil.move(old_file, new_file)
        
def rotated_reactant(target_dir, smiles_dir=DIENE_ENE_DIR, smiles_csv_name = 'all_smiles.csv', ts_name='ts', react_name='special_reactant'):
    ts_file_dir = target_dir + "/" + ts_name
    seperate_reactant_dir = target_dir + "/" + react_name
    if not os.path.isdir(seperate_reactant_dir):
        os.mkdir(seperate_reactant_dir)
    smiles_dict = pd.read_csv(smiles_dir +"/"+ smiles_csv_name).to_dict()
    smiles_idxs = list(smiles_dict["Index"].values())

    om_log_files = glob.glob(ts_file_dir + "/*.log")   
    for om_log_file in om_log_files:
        om_log = logfile_process.Logfile(om_log_file)
        start = om_log.title[4]
        title = om_log.title
        all_symbol_list = om_log.symbol_list
        all_position = om_log.running_positions[-1]
        diene_symbol_list = all_symbol_list[:start]
        diene_position = all_position[:start]

        diene_id = int(os.path.split(om_log_file)[-1].split("_")[0])
        diene_conf_id = smiles_dict['Stable_conf_id'][smiles_idxs.index(diene_id)]
        diene_mol_file = glob.glob(os.path.join(smiles_dir, "mol", f"smilesid_{diene_id:05}*.mol"))[0]
        diene_mol = Chem.MolFromMolFile(diene_mol_file, removeHs=False)
        diene_atom_lists = cycle_process.diene_atom_Idx(diene_mol, select_diene=0)
        atoma_id, atomd_id = title[0], title[1]
        for each in diene_atom_lists:
            if each[0] == atoma_id and each[-1] == atomd_id:
                atomb_id, atomc_id = each[1:3]
                break
        diene_file = glob.glob(os.path.join(smiles_dir, "mol_dft", f"smilesid_{diene_id:05}_{diene_conf_id:04}*.log"))[0]
        diene_log = logfile_process.Logfile(diene_file)
        diene_react_pisiton = diene_log.running_positions[-1]
        torsion_cos = Tool.get_torsion(diene_react_pisiton[atoma_id], diene_react_pisiton[atomb_id], diene_react_pisiton[atomc_id], diene_react_pisiton[atomd_id])
        if torsion_cos > 0:
            continue
        diene_mol = xtb_process.xtb_to_mol(diene_mol, [diene_symbol_list], [diene_position], 1)
        Chem.MolToMolFile(diene_mol, os.path.join(target_dir, "mol", os.path.split(om_log_file)[-1].split(".")[0] + "_diene.mol"))
        new_diene_file = seperate_reactant_dir + "/" + os.path.split(om_log_file)[-1].split(".")[0] + "_diene.gjf"
        format_change.block_to_gjf(symbol_list=diene_symbol_list, positions=diene_position, 
                                    file = new_diene_file, method='opt freq b3lyp/6-31g* em=gd3bj')