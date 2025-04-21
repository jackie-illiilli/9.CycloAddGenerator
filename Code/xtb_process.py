from . import format_change, cycle_process
from rdkit import Chem
from rdkit.Chem import AllChem
import shutil, glob, os
import copy

def xtb_to_mol(mol, atoms, positions, conf_limit):
    AllChem.EmbedMultipleConfs(mol, 0)
    for idx in range(conf_limit):
        conf = Chem.rdchem.Conformer(mol.GetNumAtoms())
        conf.SetId(idx)
        for i, atom in enumerate(mol.GetAtoms()):
            assert atom.GetSymbol() == atoms[idx][i]
            conf.SetAtomPosition(i, positions[idx][i])
        mol.AddConformer(conf)
    return mol

def xtb_update_mol(mol, xtbfile, conf_limit = 3, rmsd_limit=1.5):
    """read xyz and update atom positions, with xtb_dir
    parents of (read_xyz)

    Args:
        mol (Chem.Mol): 
        xtbfile (str): .xyz
        conf_limit: only read num of best result

    Returns:
        mol: Chem.Mol with new positions
    """    


    atoms, positions = read_xyz(xtbfile) 
    assert len(atoms) == len(positions)
    if conf_limit > len(atoms):
        conf_limit = len(atoms)
    first_conf = 0 
    new_mol = copy.deepcopy(mol)
    new_mol = xtb_to_mol(new_mol, atoms, positions, conf_limit)
    if conf_limit == 1:
        return new_mol
    else:
        save_conf_id = [0]
        for i in range(1, conf_limit):
            if min([AllChem.GetConformerRMS(new_mol, i, conf_id) for conf_id in save_conf_id]) > rmsd_limit:
                save_conf_id.append(i)
        atoms = [atoms[i] for i in save_conf_id]
        positions = [positions[i] for i in save_conf_id]
        mol = xtb_to_mol(mol, atoms, positions, len(save_conf_id))
    return mol

def format_constrained_atoms(rest_atoms):
    """shift 1,2,3,4,6,7,8 into 1-4, 6-8
    Args:
        rest_atoms (_type_): _description_
        all_num (_type_): _description_
    """    
    rest_atoms.append(-1)
    return_str = ''
    temp_str = ''
    for atom_id, else_atom in enumerate(rest_atoms[:-1]):
        if len(temp_str) == 0:
            temp_str += str(else_atom)
        if rest_atoms[atom_id + 1] - else_atom != 1:
            if temp_str[0] != str(else_atom):
                temp_str += "-" 
                temp_str += str(else_atom)
            if atom_id != len(rest_atoms) -2:
                temp_str += ','
            return_str += temp_str
            temp_str = ''
    return return_str

def xtb_write_xyz(mol, atom_list=None, position_list=None, xtb_dir='xtb_process/', smiles_name='test',constrain_atoms=None, dist_rest=None):
    """
    次级函数：对mol的每一个构象，在id_%.8d_%d的文件夹中生成xtb优化文件
    parents of mol_to_xyz()

    Args:
        mol (str): mol
        xtb_dir (str, optional): parnets dir of charges files. Defaults to 'xtb_process/'.
        smiles_name (str, optional): name for charge files. Defaults to 'test'.

    Returns:
        file_dir: 
    """    
    if not os.path.isdir(xtb_dir):
        os.mkdir(xtb_dir)
    xyz_files = format_change.mol_to_xyz(mol, atom_list, position_list, file_dir=xtb_dir + "%s.xyz" % smiles_name)
    file_dirs = []
    for eachfile in xyz_files:
        new_path = eachfile.split(".")[0] + "/"
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        shutil.move(eachfile, new_path + os.path.split(eachfile)[1])
        # if mol is not None:
        #     Chem.MolToMolFile(mol,new_path + "/%s.mol" % smiles_name, )
        if dist_rest is not None or constrain_atoms is not None:
            if mol is None:
                atom_num = len(atom_list)
            else:
                atom_num = mol.GetNumAtoms()
            # write_coord(new_path + '/coord.ref', atom_list, position_list[0])
            with open(new_path + '/.constrains', "wt", newline="\n") as f:
                f.write("$constrain\n")
                if constrain_atoms != None:
                    const_str = format_constrained_atoms(constrain_atoms)
                    f.write("atoms: %s\n" % const_str)
                f.write("force constant=0.5\n")
                f.write("reference=" + os.path.split(eachfile)[1] + '\n')
                if dist_rest != None:
                    for (a,b) in dist_rest:
                        f.write("distance: %d, %d, auto\n" % (a,b))
                if constrain_atoms != None:
                    else_atoms = [each for each in range(1,atom_num + 1) if each not in constrain_atoms]
                    else_str = format_constrained_atoms(else_atoms)
                    f.write("$metadyn\natoms: ")
                    f.write(else_str + '\n')
                f.write("$end\n")
        file_dirs.append(new_path + os.path.split(eachfile)[1])
    return file_dirs

def write_xtb_pbs(pbs_path, charge, que="epsilon", root_dir='charg_0'):
    """xtb运行的shell脚本生成工具

    Args:
        xyz_name (_type_): _description_
        charge (_type_): _description_
        que (str, optional): _description_. Defaults to "epsilon".
        root_dir (str, optional): _description_. Defaults to 'charg_0'.
    """    
    with open(pbs_path, "wt", newline="\n") as f:
        f.write("#!/bin/bash\n#PBS -l nodes=1:ppn=28\n#PBS -l walltime=99:00:00\n#PBS -N crest-\n#PBS -q %s\n#PBS -j oe\n#PBS -o jobID.$PBS_JOBID\n" % que)
        f.write("export OMP_NUM_THREADS=28\nexport MKL_NUM_THREADS=28\nexport OMP_STACKSIZE=6000m\n\n")
        f.write("cd $PBS_O_WORKDIR\ntouch jobID.$PBS_JOBID\nrootdir=%s\n" % root_dir)
        f.write("folders=`ls $PBS_O_WORKDIR/$rootdir/`\nfor folder in $folders\ndo\n    cd $PBS_O_WORKDIR/$rootdir/$folder\n")
        f.write("    crest $folder.xyz -T 28 -gfn2 -chrg %d -uhf 0 -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\n" % charge)
        f.write("done")

def xtb_main(smiles_names, smileses, dir_path='xtb_process', que="epsilon", core=1):
    """    !! 主流程
    对于smileses的所有分子，按照电荷进行归类，按照文件夹名称进行处理

    Args:
        smileses (_type_): smiles or mols
        dir_path (str, optional): root dir saved charge files. Defaults to 'xtb_process'.
        que (str, optional): 超算队列. Defaults to "epsilon".
        core(int): 是否要分成多个文件夹计算
    """    

    if os.path.isdir(dir_path):
        # raise ValueError("Path has existed")
        pass
    else:
        os.mkdir(dir_path)
    charges = set()
    core_size = 0
    core_id = 0
    each_size = len(smileses) // core + 1
    for i, (smiles_name, smiles) in enumerate(zip(smiles_names, smileses)):
        if type(smiles) == str:
            mol = cycle_process.smiles2mol(smiles)
            if mol is None:
                print(smiles)
        elif type(smiles) == Chem.Mol:
            mol = smiles
        charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
        charges.add(charge)
        if charge == 0:
            core_size +=1
            now_core_id = core_id
        else:
            now_core_id = 0
        if core_size >= each_size:
            core_id += 1
            core_size = 0

        # AllChem.MMFFOptimizeMoleculeConfs(mol)
        xtb_write_xyz(mol, xtb_dir=dir_path + "/" + "charg_%d_%d/" % (charge, now_core_id), smiles_name=smiles_name)
    with open(dir_path + "/suball", "wt", newline='\n') as f:
        for eachcharge in charges:
            if eachcharge == 0:
                for now_core_id in range(core_id + 1):
                    pbs_path = dir_path + "/" + "xtb_%d_%d.pbs" % (eachcharge, now_core_id)
                    write_xtb_pbs(pbs_path, eachcharge, que, root_dir="charg_%d_%d" % (eachcharge, now_core_id))
                    f.write("qsub %s\n" % "xtb_%d_%d.pbs" % (eachcharge, now_core_id))
            else:
                pbs_path = dir_path + "/" + "xtb_%d_%d.pbs" % (eachcharge, 0)
                write_xtb_pbs(pbs_path, eachcharge, que, root_dir="charg_%d_%d" % (eachcharge, 0))
                f.write("qsub %s\n" % "xtb_%d_%d.pbs" % (eachcharge, 0))

def xtb_om(all_result, dir_path='xtb_process', que="epsilon", core=1):
    """    !! 主流程2
    对于smileses的所有分子，按照10个一批

    Args:
        all_result (list): [id, title, symbol_list, new_position, charge, constrain_atoms, dist_const]
        dir_path (str, optional): root dir saved charge files. Defaults to 'xtb_process'.
        que (str, optional): 超算队列. Defaults to "epsilon".
    """    

    if os.path.isdir(dir_path):
        # raise ValueError("Path has existed")
        pass
    else:
        os.mkdir(dir_path)
    all_charge = set()
    core_size = 0
    core_id = 0
    each_size = len(all_result) // core + 1
    for (smiles_name, title, symbol_list, new_position, charge, constrain_atoms, dist_const) in all_result:
        all_charge.add(charge)

        if charge == 0:
            core_size +=1
            now_core_id = core_id
        else:
            now_core_id = 0
        if core_size > each_size:
            core_id += 1
            core_size = 0

        new_position = [new_position]
        xtb_write_xyz(mol=None, atom_list=symbol_list, position_list=new_position, xtb_dir=dir_path + "/" + "charg_%d_%d/" % (charge, now_core_id) , smiles_name=smiles_name, constrain_atoms=constrain_atoms, dist_rest=dist_const)
    with open(dir_path + "/suball", "wt", newline='\n') as f:
        for eachcharge in all_charge:
            if eachcharge == 0:
                for now_core_id in range(core_id + 1):
                    pbs_path = dir_path + "/" + "xtb_%d_%d.pbs" % (eachcharge, now_core_id)
                    write_xtb_pbs(pbs_path, eachcharge, que, root_dir="charg_%d_%d" % (eachcharge, now_core_id))
                    f.write("qsub %s\n" % "xtb_%d_%d.pbs" % (eachcharge, now_core_id))
            else:
                pbs_path = dir_path + "/" + "xtb_%d_%d.pbs" % (eachcharge, 0)
                write_xtb_pbs(pbs_path, eachcharge, que, root_dir="charg_%d_%d" % (eachcharge, 0))
                f.write("qsub %s\n" % "xtb_%d_%d.pbs" % (eachcharge, 0))

def check_xtb_normal(root_dir):
    error_code = 0
    wait_check_dir = glob.glob(root_dir + "/*/*")
    for each_file in wait_check_dir:
        with open(each_file + "/crest.out", "rt") as f:
            lines = f.readlines()
        if not lines[-1].startswith(" CREST terminated normally."):
            print(each_file.split('/')[-1], "not end normally")
            error_code = 1
    if not error_code:
        print("All End Normally!")
        return 1
    else:
        return 0

def read_xyz(file_dir):
    """read .zyx files, can read multiple molecules
    sub of read_xyz
    Args:
        file_dir (str): file_dir endwith .xyz

    Returns:
        atoms: list of atoms
        position: array of position
    """    
    with open(file_dir) as f:
        lines = f.readlines()
    line1 = lines[0]
    mol_num = len([line for line in lines if line == line1])
    atom_list = []
    positions = []
    start_id = 0
    for eachmol in range(mol_num):
        atom_num = int(lines[start_id].strip("\n").split("  ")[-1])
        if atom_num == 0:
            raise TypeError("Not a real xyz")
        atoms = []
        position = []
        start_id += 2
        for i in range(atom_num):
            pro_line = lines[start_id].split()
            atoms.append(pro_line[0])
            position.append([float(pro_line[1]), float(pro_line[2]), float(pro_line[3])])
            start_id += 1
        assert len(atoms) == atom_num
        atom_list.append(atoms)
        positions.append(position)
    return atom_list, positions

def xtb_is_success(xtb_dir):
    if os.path.isfile(xtb_dir + '/crest.out'):
        with open(xtb_dir + '/crest.out', "rt", encoding='UTF-8') as f:
            final_line = f.readlines()[-1]
        if final_line.startswith(" CREST terminated normally."):
            return True
    print(xtb_dir, "  unsuccessful!")
    return False

def after_xtb(mol, 
            root_dir="xtb_process", 
            save_dir="xtb_result", 
            mol_str=0, 
            conf_limit=3,
            rmsd_limit=1.5,
            xtb_title=None, 
            method="opt freq b3lyp/6-31g* em=gd3bj g09def", 
            freeze=[]):
    """根据root_dir, charge, mol 更新优化分子

    Args:
        mol (_type_): _description_
        mol_id (str): _description_
        root_dir (str, optional): _description_. Defaults to "xtb_process".
        charge(int, optional):charge for xtbdir
    """  
    xtb_dir = glob.glob(root_dir + "/" + "%s_*" % mol_str)
    if len(xtb_dir) == 0:
        print("mol_str:%sdidn't find crest_best!" % (mol_str))
        return 0
    charge = int(xtb_dir[0].split("\\")[-2].split("_")[-2])
    for conf_id, each_dir in enumerate(xtb_dir):
        if not os.path.isfile(each_dir + "/crest_best.xyz"):
            print("mol_str:%s, conf_id:%d didn't find crest_best!" % (mol_str, conf_id))
            # xtb_files = glob.glob(each_dir + '/id*.xyz')
            # xtb_update_mol(mol, xtb_files[0])
        else:
            try:
                xtb_update_mol(mol, each_dir + "/crest_conformers.xyz", conf_limit, rmsd_limit)
            except:
                print("mol_str:%s, conf_id:%d have something wrong!" % (mol_str, conf_id))
    for conf_id in range(len(mol.GetConformers())):
        file_dir = save_dir + "/%s_%.4d.gjf" % (mol_str, conf_id)
        if len(freeze):
            format_change.mol_to_gjf(mol, file_dir, confid=conf_id, method=method, charge=charge, title=xtb_title, freeze=freeze)
        else:
            format_change.mol_to_gjf(mol, file_dir, confid=conf_id, method=method, charge=charge, title=xtb_title)

# 检查错误






def find_min_eng_log(opt_str="xtb_result/id_000101*.log", limit=1, return_eng = False):
    pre_opt_files = glob.glob(opt_str)
    opt_files = []
    eng_files = []
    for opt_file in pre_opt_files:
        try:
            eng_file = opt_file.split(".log")[0] + "_eng.log"
            eng_file = os.path.split(eng_file)[0] + "_eng" + "/" + os.path.split(eng_file)[1]
            assert os.path.isfile(eng_file)
            eng_files.append(eng_file)
            opt_files.append(opt_file)
        except:
            pass
    result_index = find_min_eng_log_(eng_files, opt_files, limit, return_eng)
    return result_index

def find_min_eng_log_(eng_files, opt_files, limit, return_eng=False):
    gbs_engs = []
    assert len(eng_files) == len(opt_files)
    for eng_file, opt_file in zip(eng_files, opt_files):
        gbs_cor = float(cycle_process.read_log_eng(opt_file)[-1])
        ee = float(cycle_process.read_log_eng(eng_file)[0])
        gbs_engs.append(gbs_cor + ee)
    min_conf = np.array(gbs_engs).argsort()
    min_opt_files = [opt_files[each] for each in min_conf][:limit]
    # print(gbs_engs)
    if return_eng:
        return min_opt_files,[gbs_engs[each] for each in min_conf][:limit]
    else:
        return min_opt_files

def write_coord(file_dir, symbol_list, positions):
    with open(file_dir, 'wt')as f:
        symbol_list = [s.lower() for s in symbol_list]
        f.write("$coord\n")
        for symbol, pos in zip(symbol_list, positions):
            f.write("    %.12f    %.12f    %.12f   %s\n" % (pos[0], pos[1], pos[2], symbol))
        f.write("$end\n")

def shift_to_sugan(target_file, quene_id = 1):
    pbs_files = glob.glob(target_file + '/*.pbs')
    for pbs_file in pbs_files:
        with open(pbs_file, "wt",  newline='\n') as f:
            number = int(pbs_file.split(".pbs")[0].split("_")[-1])
            f.write("#!/bin/bash\n#SBATCH -J g16\n#SBATCH -N 1\n#SBATCH --ntasks-per-node=32\n#SBATCH -p hfacnormal%.2d\n\nroot=`pwd`\nrootdir=charg_0_%d\nfolders=`ls $root/$rootdir/`\n" % (quene_id, number))
            f.write("for folder in $folders\ndo\n    cd $root/$rootdir/$folder\n    crest $folder.xyz -T 28 -gfn2 -chrg 0 -uhf 0 -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\ndone\n")
    with open(target_file + '/suball', "wt",  newline='\n') as f:
        for pbs_file in pbs_files:
            name = os.path.split(pbs_file)[-1]
            f.write("sbatch %s\n" % name)

def shift_to_parra(target_file):
    pbs_files = glob.glob(target_file + '/*.pbs')
    for pbs_file in pbs_files:
        with open(pbs_file, "wt",  newline='\n') as f:
            number = int(pbs_file.split(".pbs")[0].split("_")[-1])
            f.write("#!/bin/bash\n#SBATCH -p amd_512\n#SBATCH -N 1\n#SBATCH -n 1\n#SBATCH -c 28\n\nroot=`pwd`\nrootdir=charg_0_%d\nfolders=`ls $root/$rootdir/`\n" % (number))
            f.write("for folder in $folders\ndo\n    cd $root/$rootdir/$folder\n    crest $folder.xyz -T 28 -gfn2 -chrg 0 -uhf 0 -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\ndone\n")
    with open(target_file + '/suball', "wt",  newline='\n') as f:
        for pbs_file in pbs_files:
            name = os.path.split(pbs_file)[-1]
            f.write("sbatch %s\n" % name)