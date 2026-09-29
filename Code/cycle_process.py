from . import format_change, xtb_process, Tool, logfile_process

import os, copy
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import AssignStereochemistryFrom3D
from rdkit.Geometry import Point3D


def mol_add_Hs(molfile):
    """Add hydrogens to a molfile structure.

    Args:
        molfile (_type_): _description_
    """    
    mol = Chem.MolFromMolFile(molfile)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    Chem.MolToMolFile(mol, molfile)


# def group_atoms(coordinates, threshold):
#     """
#     Group atoms by spatial proximity.
    
#     Args:
#     - coordinates: N × 3 NumPy array of atom coordinates.
#     - threshold: Distance threshold for grouping atoms.
    
#     Returns:
#     - List of atom-index groups.
#     """
#     # Build a KD-tree.
#     from scipy.spatial import KDTree
#     tree = KDTree(coordinates)

#     # Find neighboring points.
#     groups = []
#     visited = set()
#     for i in range(len(coordinates)):
#         if i in visited:
#             continue
#         group = []
#         queue = [i]
#         while queue:
#             j = queue.pop(0)
#             if j in visited:
#                 continue
#             visited.add(j)
#             group.append(j)
#             neighbors = tree.query_ball_point(coordinates[j], threshold)
#             queue.extend([k for k in neighbors if k not in visited])
#         groups.append(group)

#     return groups

def is_diene(smiles, kekulize=True):
    """Check for a Diels–Alder-eligible double–single–double motif.

    Reject motifs where a ring contains the central bond and only one adjacent double bond.

    Args:
        smiles (str): SMILES

    Returns:
        Bool: whether can be a 4π
    """    
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if kekulize:
        Chem.Kekulize(mol)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            center_id = bond.GetIdx()
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            pre_atom_idx = [atom1.GetIdx(), atom2.GetIdx()]
            bonds1 = [bond for bond in atom1.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
            bonds2 = [bond for bond in atom2.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
            if len(bonds1) == 1 and len(bonds2) == 1:
                bond1, bond2 = bonds1[-1], bonds2[-1]
                atoms1 = [bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()]
                atoms2 = [bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx()]
                atom0 = [atom_id for atom_id in atoms1 if atom_id not in pre_atom_idx][0]
                atom3 = [atom_id for atom_id in atoms2 if atom_id not in pre_atom_idx][0]
                diene_list = [atom0, atom1.GetIdx(), atom2.GetIdx(), atom3]
                try:
                # check enes are not in different ring
                    [a1, a2, b1, b2] = diene_list
                    judge = True
                    for eachring in Chem.GetSymmSSSR(mol):
                        if a1 in eachring and a2 in eachring and b1 in eachring:
                            if b2 not in eachring:
                                judge = False
                        if a2 in eachring and b2 in eachring and b1 in eachring:
                            if a1 not in eachring:
                                judge = False
                    if judge == True:
                        return True
                    return False
                except:
                    return False
    return False
    
def find_all_diene(mol):
    """Find atom-index lists matching the 2-1-2 pattern in a molecule.
    Legacy atom-pattern matcher; retained for compatibility.

    Args:
        mol (_type_): _description_

    Returns:
        list: _description_
    """    
    Chem.Kekulize(mol)
    return_list = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            center_id = bond.GetIdx()
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            pre_atom_idx = [atom1.GetIdx(), atom2.GetIdx()]
            bonds1 = [bond for bond in atom1.GetBonds(
            ) if bond.GetBondType() == Chem.BondType.DOUBLE]
            bonds2 = [bond for bond in atom2.GetBonds(
            ) if bond.GetBondType() == Chem.BondType.DOUBLE]
            if len(bonds1) > 0 and len(bonds2) > 0:
                # assert len(bonds1) == 1 and len(bonds2) == 1 # don't consider —NO2-, —SO3-,
                bond1, bond2 = bonds1[-1], bonds2[-1]
                atoms1 = [bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()]
                atoms2 = [bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx()]
                atom0 = [
                    atomid for atomid in atoms1 if atomid not in pre_atom_idx][0]
                atom3 = [
                    atomid for atomid in atoms2 if atomid not in pre_atom_idx][0]
                # atoms = [atomid for atomid in atoms if atomid not in pre_atom_idx]
                atoms = [atom0, atom1.GetIdx(), atom2.GetIdx(), atom3]
                # try:
                # check enes are not in different ring
                [a1, a2, b1, b2] = atoms
                judge = True
                for eachring in Chem.GetSymmSSSR(mol):
                    if a1 in eachring and a2 in eachring and b1 in eachring:
                        if b2 not in eachring:
                            judge = False
                    if a2 in eachring and b2 in eachring and b1 in eachring:
                        if a1 not in eachring:
                            judge = False
                if judge == True:
                    return_list.append(atoms)
                # except:
                #     pass
            
    return Tool.remove_same(return_list)

def dieno_atom_Idx(mol):
    """Find atom-index lists containing double or triple bonds.
    If a C=C bond is present, ignore heteroatom double bonds and aromatic bonds.

    Args:
        mol (_type_): _description_

    Returns:
        _type_: _description_
    """    
    temp_mol = mol
    Chem.Kekulize(temp_mol)
    atom_list, select_bonds = [], []
    for bond in temp_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            select_bonds.append(bond)
        if bond.GetBondType() == Chem.BondType.TRIPLE:
            select_bonds.append(bond)
    Chem.SanitizeMol(temp_mol)
    for begin_atom, final_atom, bond in [[bond.GetBeginAtom(), bond.GetEndAtom(), bond] for bond in select_bonds]:
        if (begin_atom.GetSymbol() == "C" and final_atom.GetSymbol() == "C") and (bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE):
            atom_list.append([begin_atom.GetIdx(), final_atom.GetIdx()])
    if len(atom_list) == 0:
        atom_list = ([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                     for bond in select_bonds])
    return atom_list

def diene_atom_Idx(mol, select_diene=0):
    """Rank candidate reactive sites using empirical rules.

    Args:
        mol (_type_): _description_
        select_diene (int, optional): 0: no filtering; 100: carbon termini; 200: aromatic motifs with carbon termini and a central nitrogen. Defaults to 0.

    Returns:
        _type_: _description_
    """    
    all_diene = find_all_diene(mol)

    if int(select_diene) <= 0 or len(all_diene) <= 1:
        return all_diene
    else:
        Chem.SanitizeMol(mol)
        select_score = [[0, each] for each in all_diene]
        for i, diene_index in enumerate(all_diene):
            a1, a2, b1, b2 = diene_index
            bonda = mol.GetBondBetweenAtoms(a1, a2)
            bondb = mol.GetBondBetweenAtoms(b1, b2)
            if bonda.GetBondType() == Chem.BondType.DOUBLE:
                select_score[i][0] += 10
            if bondb.GetBondType() == Chem.BondType.DOUBLE:
                select_score[i][0] += 10
            if mol.GetAtomWithIdx(a1).GetSymbol() == "C":
                select_score[i][0] += 50
            if mol.GetAtomWithIdx(b2).GetSymbol() == "C":
                select_score[i][0] += 50
            for ring in Chem.GetSymmSSSR(mol):
                if len([1 for each in diene_index if each in list(ring)]) == 4:
                    if mol.GetAtomWithIdx(a2).GetSymbol() == "N":
                        select_score[i][0] += 50
                    if mol.GetAtomWithIdx(b1).GetSymbol() == "N":
                        select_score[i][0] += 50

        if int(select_diene) > 10:
            return_list = [each for i,
                           each in select_score if i >= int(select_diene)]
            if len(return_list) != 0:
                return return_list
            else:
                return all_diene
        else:
            return select_score[:int(select_diene)]

def move(a):
    """Convert a 3D translation vector to a 4 × 4 homogeneous transformation matrix.

    Args:
        a (iterable): 

    Returns:
        array: size = 4*4
    """    
    x, y, z = a[:3]
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def rotation(a, sin, cos):
    """Build a 4 × 4 matrix for rotation about axis a using the specified sine and cosine.

    Args:
        a (array): 3D vector.
        sin (int): 
        cos (int): 

    Returns:
        array: size 4*4
    """    
    a = np.array(a)
    a = a / np.sqrt(a @ a.T)
    u, v, w = a
    return np.array([[u * u + (1 - u * u) * cos, u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v * sin, 0],
                     [u * v * (1 - cos) + w * sin, v * v + (1 - v * v)
                      * cos, v * w * (1 - cos) - u * sin, 0],
                     [u * w * (1 - cos) - v * sin, v * w * (1 - cos) +
                      u * sin, w * w + (1 - w * w) * cos, 0],
                     [0, 0, 0, 1]])

def trfm_rot(a, b, c, position=[], center_point=np.array([0, 0, 0])):
    """Transform coordinates so a–b lies on x, a and b are symmetric about center_point, and c lies in the xy plane.

    Args:
        a (_type_): 3D coordinate; transformed to the negative x direction.
        b (_type_): 3D coordinate; transformed to the positive x direction.
        c (_type_): 3D coordinate; transformed to positive y with z set to zero.
        position (list, optional): 3D or 4D coordinate array. Defaults to [].
        center_point (_type_, optional): _description_. Defaults to np.array([0, 0, 0]).

    Returns:
        array: new array (4*4)
        or tr_array, rot1_array, rot2_array, tr_array2
    """    
    zero_point = np.array([0, 0, 0])
    x, y, z = np.array(a), np.array(b), np.array(c)
    x_axis = np.array([1, 0, 0])
    mid_point = (x + y)/2
    tr_array_3d = zero_point - mid_point
    tr_array = move(tr_array_3d)
    x += tr_array_3d
    y += tr_array_3d
    z += tr_array_3d
    tr_array2 = move(center_point - zero_point)
    xy = y - x
    xy = xy / np.sqrt((xy @ xy))
    law_axis = np.cross(xy, x_axis)
    if (law_axis == np.array([0, 0, 0])).all() == False:
        law_axis /= np.sqrt((law_axis @ law_axis))
        sin = np.sqrt(xy[1] * xy[1] + xy[2] * xy[2])
        cos = xy[0]
        rot1_array = rotation(law_axis, sin, cos)
    else:
        rot1_array = np.eye(4)
    oz = z - zero_point
    oz = rot1_array[:3, :3] @ oz
    oz[0] = 0
    if (oz == np.array([0, 0, 0])).all() == False:
        oz = oz / np.sqrt((oz @ oz))
    sin = oz[2]
    cos = oz[1]
    rot2_array = rotation(-1 * x_axis, sin, cos)
    if len(position) == 0:
        return tr_array, rot1_array, rot2_array, tr_array2
    else:
        if len(position[0]) == 3:
            up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        else:
            up_position = position.T
        up_position = tr_array @ up_position
        up_position = rot1_array @ up_position
        up_position = rot2_array @ up_position
        up_position = tr_array2 @ up_position
        up_position = up_position.T
    return up_position


def diene_cut_down(mol, center_atoms):
    """Split the diene across its central single bond to allow cis/trans rotation.
    Args:
        mol (mol): _description_
        center_atoms (list): _description_

    Returns:
        list,list: list AB
    """    
    for eachring in Chem.GetSymmSSSR(mol):
        assert sum([atomid in eachring for atomid in center_atoms]
                   ) != len(center_atoms)
        # Assert atoms in two ene not in a ring, or the cut_down will fail.
    if "H" not in [atom.GetSymbol() for atom in mol.GetAtoms()]:
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
    left_atoms_set, right_atoms_set = set(), set()
    temp_left_set, temp_right_set = set(), set()
    for each in center_atoms[:2]:
        left_atoms_set.add(each)
        temp_left_set.add(each)
    for each in center_atoms[2:]:
        right_atoms_set.add(each)
        temp_right_set.add(each)
    while(True):
        new_left_set = set()  # Save new atom in every epochs
        for eachatomnum in temp_left_set:
            Atom = mol.GetAtomWithIdx(eachatomnum)
            neighbors = set([atom.GetIdx() for atom in Atom.GetNeighbors() if atom.GetIdx(
            ) not in left_atoms_set and atom.GetIdx() not in right_atoms_set])
            new_left_set = new_left_set | neighbors
            left_atoms_set = left_atoms_set | new_left_set
        temp_left_set = new_left_set
        if len(temp_left_set) == 0:
            break

    while(True):
        new_right_set = set()
        for eachatomnum in temp_right_set:
            Atom = mol.GetAtomWithIdx(eachatomnum)
            neighbors = set([atom.GetIdx() for atom in Atom.GetNeighbors() if atom.GetIdx(
            ) not in left_atoms_set and atom.GetIdx() not in right_atoms_set])
            new_right_set = new_right_set | neighbors
            right_atoms_set = right_atoms_set | new_right_set
        temp_right_set = new_right_set
        if len(temp_right_set) == 0:
            break

    return left_atoms_set, right_atoms_set

def trans_to_cis(mol, position, center_atoms):
    """Convert a trans diene geometry to cis by rotating one half.

    Args:
        mol (_type_): _description_
        center_atoms (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # atomA, atomB, atomC, atomD = center_atoms
    A, B, C, D = [np.array(position[each]) for each in center_atoms]
    cos0 = Tool.get_torsion(A, B, C, D)
    if cos0 > 0:  # No rotation is required.
        return position, 0
    else:
        try:
            left_atoms_set, right_atoms_set = diene_cut_down(mol, center_atoms)
            # Move atom B to the origin; vector BC then defines the rotation axis.
            move_array = np.zeros(3) - B
            mol = move_mol(mol, move_array)
            conformer = mol.GetConformers()[0]
            position = conformer.GetPositions()
            left_posi = np.array(
                [each for i, each in enumerate(position) if i in left_atoms_set])
            A, B, C, D = [np.array(position[each]) for each in center_atoms]
            axis = C
            up_position = np.insert(left_posi, 3, np.ones(len(left_posi)), 1)
            rot_matrix = rotation(axis, sin=0.174, cos=-0.985)
            up_position = (rot_matrix @ up_position.T).T
            for i, c in zip(left_atoms_set, up_position):
                conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
            # AllChem.EmbedMolecule(mol)
            position = conformer.GetPositions()
        except:
            return position, 0
    return position, 1

def change_position(react1, prop="diene", name=None, save=False, addHs=True, select_diene=0, return_tran_cis=False):
    """Orient the diene or ene with the alkene plane in xy, the reactive-bond midpoint at the origin, and the reactive axis along x.

    Args:
        react1 (_type_): _description_
        prop (str, optional): _description_. Defaults to "diene".
        name (_type_, optional): _description_. Defaults to None.
        save (bool, optional): _description_. Defaults to False.
        addHs (bool, optional): _description_. Defaults to True.
        select_diene (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """    
    mol = copy.deepcopy(react1)
    if addHs == True and "H" not in [atom.GetSymbol() for atom in mol.GetAtoms()]:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
    return_mols = []
    assert prop == "diene" or "dieno" or "dipole"
    if name == None:
        name = "default"
    if prop == "diene":
        center_atoms_group = diene_atom_Idx(mol, select_diene=select_diene)
    else:
        center_atoms_group = dieno_atom_Idx(mol)
    # AllChem.EmbedMultipleConfs(mol, numConfs=50, maxAttempts=100, )
    for i, center_atoms in enumerate(center_atoms_group):
        return_mol = copy.deepcopy(mol)
        for conformer in return_mol.GetConformers():
            try:
                position = conformer.GetPositions()
                Chem.Kekulize(return_mol)
                if prop == "diene":
                    position, tran_cis = trans_to_cis(return_mol, position, center_atoms)
                    new_position = trfm_rot(
                        position[center_atoms[0]], position[center_atoms[3]], position[center_atoms[1]], position)[:, :3]
                else:
                    tran_cis = 0
                    another_bond = [bond for bond in return_mol.GetAtomWithIdx(
                        center_atoms[0]).GetBonds() if bond.GetIdx() != return_mol.GetBondBetweenAtoms(center_atoms[0], center_atoms[1]).GetIdx()]
                    if len(another_bond) == 0:
                        another_bond = [bond for bond in return_mol.GetAtomWithIdx(
                            center_atoms[1]).GetBonds() if bond.GetIdx() != return_mol.GetBondBetweenAtoms(center_atoms[0], center_atoms[1]).GetIdx()]
                    another_bond = another_bond[-1]
                    another_atom = [atomid for atomid in [another_bond.GetEndAtomIdx(
                    ), another_bond.GetBeginAtomIdx()] if atomid != center_atoms[0]]
                    new_position = trfm_rot(
                        position[center_atoms[0]], position[center_atoms[1]], position[another_atom[0]], position)[:, :3]
            except:
                continue
            Chem.SanitizeMol(return_mol)

            # bposition = np.insert(position, 3, np.ones(len(position)), 1)
            # center_position = (bposition[center_atoms[0]] + bposition[center_atoms[1]])/2

            for i, c in enumerate(new_position):
                conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
        if return_tran_cis:
            return_mols.append([return_mol, center_atoms, tran_cis])
        else:
            return_mols.append([return_mol, center_atoms])
    return return_mols

def rot_mol(mol, axis=np.array([0, 1, 0]), sin=0, cos=-1):
    """Rotate every conformer; the default is 180° about the y-axis.

    Args:
        mol (_type_): _description_
        axis (array, optional): Rotation axis. Defaults to np.array([0, 1, 0]).
        sin (int, optional): _description_. Defaults to 0.
        cos (int, optional): _description_. Defaults to -1.

    Returns:
        mol: _description_
    """    
    # Define rotation corner is 180 degree
    react1 = copy.deepcopy(mol)
    for conformer in react1.GetConformers():
        position = conformer.GetPositions()
        up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        matrix = rotation(axis, sin, cos)
        up_position = (matrix @ up_position).T
        for i, c in enumerate(up_position):
            conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
    return react1


def move_mol(mol, array=np.array([0, 0, 1.5])):
    """Translate every conformer; the default ene displacement is 1.5 units along z.

    Args:
        mol (_type_): _description_
        array (array, optional): _description_. Defaults to np.array([0, 0, 1.5]).

    Returns:
        _type_: _description_
    """    
    react1 = copy.deepcopy(mol)
    for conformer in react1.GetConformers():
        position = conformer.GetPositions()
        up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        matrix = move(array)
        up_position = (matrix @ up_position).T
        for i, c in enumerate(up_position):
            conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
    return react1

def check_double_bond_ZE(mol, old_position, new_position, ignore=[], ignore_H=True):
    is_error = []
    Chem.Kekulize(mol)
    for bond in mol.GetBonds():
        # if bond.GetStereo() == Chem.BondStereo.STEREONONE:
        #     continue
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        continue_num = 0
        for each_ignore in ignore:
            if atom1.GetIdx() in each_ignore and atom2.GetIdx() in each_ignore:
                continue_num = 1
        if continue_num:
            continue
        if atom1.GetSymbol() not in ["C", "N"] or atom2.GetSymbol() not in ["C", "N"]:
            continue
        if ignore_H:
            a_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atom1.GetIdx()).GetNeighbors() if each.GetIdx() != atom2.GetIdx() and each.GetAtomicNum() != 1]
            b_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atom2.GetIdx()).GetNeighbors() if each.GetIdx() != atom1.GetIdx() and each.GetAtomicNum() != 1]
        else:
            a_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atom1.GetIdx()).GetNeighbors() if each.GetIdx() != atom2.GetIdx()]
            b_neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(atom2.GetIdx()).GetNeighbors() if each.GetIdx() != atom1.GetIdx()]
        
        if len(a_neighbor) <= 0 or len(b_neighbor) <= 0:
            continue
        a_nei_id = a_neighbor[0]
        b_nei_id = b_neighbor[0]
        old_cos = Tool.get_torsion(old_position[a_nei_id], old_position[atom1.GetIdx()], old_position[atom2.GetIdx()], old_position[b_nei_id])
        if abs(old_cos) < 0.3:
            continue
        new_cos = Tool.get_torsion(new_position[a_nei_id], new_position[atom1.GetIdx()], new_position[atom2.GetIdx()], new_position[b_nei_id])
        if (old_cos - 0.1736) * (new_cos - 0.1736) < 0:
            is_error = [a_nei_id, atom1.GetIdx(), atom2.GetIdx(), b_nei_id]
            break
    return is_error

def check_special_bond_ZE(mol, old_position, new_position, atom_ids=[]):
    is_error = []
    sum_atoms = atom_ids[0] + atom_ids[-1]
    for ring in Chem.GetSymmSSSR(mol):
        ring = list(ring)
        if sum([1 for each in sum_atoms if each in ring]) == 4:
            return is_error

    for [atom_a_id, atom_b_id], value in zip(atom_ids, [0, 1]):
        atom1 = mol.GetAtomWithIdx(atom_a_id)
        atom2 = mol.GetAtomWithIdx(atom_b_id)
        if atom1.GetSymbol() != "C" or atom2.GetSymbol() != "C":
            continue
        a_neighbor = [each.GetIdx() for each in atom1.GetNeighbors() if each.GetIdx() != atom2.GetIdx() and each.GetAtomicNum() != 1]
        b_neighbor = [each.GetIdx() for each in atom2.GetNeighbors() if each.GetIdx() != atom1.GetIdx() and each.GetAtomicNum() != 1]
        if len(a_neighbor) <= 0 or len(b_neighbor) <= 0:
            continue
        a_nei_id = a_neighbor[0]
        b_nei_id = b_neighbor[0]
        old_cos = Tool.get_torsion(old_position[a_nei_id], old_position[atom1.GetIdx()], old_position[atom2.GetIdx()], old_position[b_nei_id])
        if abs(old_cos) < 0.3:
            continue
        BA = new_position[a_nei_id] - new_position[atom1.GetIdx()]
        BC = new_position[atom2.GetIdx()] - new_position[atom1.GetIdx()]
        CD = new_position[b_nei_id] - new_position[atom2.GetIdx()]
        cross = np.cross(BA, BC)
        new_sin = Tool.get_array_cos(cross, CD)
        if abs(new_sin) > 0.1:
            if value == 0:
                result = (new_sin * old_cos < 0)
            else:
                result = (new_sin * old_cos > 0)
            if result:
                is_error = [a_nei_id, atom1.GetIdx(), atom2.GetIdx(), b_nei_id]
                break
    return is_error
        
def up_down_check(position, diene_num1, diene_num2, dieno_num1, start, diene_mol):
    diene_point1 = position[diene_num1]
    diene_point2 = position[diene_num2]
    dieno_point1 = position[dieno_num1 + start]
    diene_atom_lists = diene_atom_Idx(diene_mol, select_diene=0)
    for each in diene_atom_lists:
        if each[0] == diene_num1 and each[-1] == diene_num2:
            atomb_id, atomc_id = each[1:3]
            break
    diene_point3 = position[atomb_id]
    # print(diene_point3)
    new_position = trfm_rot(diene_point1, diene_point2, diene_point3, position)
    dieno_point1 = new_position[dieno_num1 + start][:3]
    if dieno_point1[2] < 0:
        return False
    else:
        return True


def comb_mol(diene, dieno, diene_list, dieno_list, prop="DA", FF_opt = True):
    """Build a Diels–Alder product from the specified atoms, optionally optimize it with a force field, and calculate its energy.

    Args:
        diene (mol): _description_
        dieno (mol): _description_
        diene_list (list): a=b-c=d
        dieno_list (list): a=b

    Returns:
        mol: _description_
    """    
    diene_num1, diene_num2, diene_num3, diene_num4 = diene_list
    dieno_num1, dieno_num2 = dieno_list
    start = diene.GetNumAtoms()
    comb = Chem.CombineMols(diene, dieno)
    rwcomb = Chem.RWMol(comb)
    Chem.Kekulize(rwcomb)
    another_bond1 = rwcomb.GetBondBetweenAtoms(diene_num1, diene_num2)
    another_bond2 = rwcomb.GetBondBetweenAtoms(diene_num3, diene_num4)
    bond3 = rwcomb.GetBondBetweenAtoms(diene_num2, diene_num3)
    rwcomb.AddBond(diene_num1, dieno_num1 + start, Chem.BondType.SINGLE)
    rwcomb.AddBond(diene_num4, dieno_num2 + start, Chem.BondType.SINGLE)
    another_bond1.SetBondType(Chem.BondType.SINGLE)
    another_bond2.SetBondType(Chem.BondType.SINGLE)
    bond4 = rwcomb.GetBondBetweenAtoms(dieno_num1 + start, dieno_num2 + start)
    bond3.SetBondType(Chem.BondType.DOUBLE)
    if bond4.GetBondType() == Chem.BondType.DOUBLE:
        bond4.SetBondType(Chem.BondType.SINGLE)
    elif bond4.GetBondType() == Chem.BondType.TRIPLE:
        bond4.SetBondType(Chem.BondType.DOUBLE)
    else:
        print(bond4.GetBondType())
        raise TypeError("Bond Type is Error")
    Chem.SanitizeMol(rwcomb)
    new_comb = rwcomb.GetMol()
    new_mol = copy.deepcopy(new_comb)
    new_conf_id = 0
    is_error = [1]
    if not FF_opt:
        return new_mol, 0
    Chem.AllChem.EmbedMultipleConfs(new_mol, 0)
    for conf_id, _ in enumerate(new_comb.GetConformers()):
        try:
            AllChem.MMFFOptimizeMolecule(new_comb, confId=conf_id)
            Chem.rdForceFieldHelpers.UFFOptimizeMolecule(new_comb, confId=conf_id)
            conformer = new_comb.GetConformer(conf_id)
            position = conformer.GetPositions()
            new_diene_position = position[:start]
            new_dieno_position = position[start:]
            old_diene_position = diene.GetConformers()[0].GetPositions()
            old_dieno_position = dieno.GetConformers()[0].GetPositions()
            is_error = check_double_bond_ZE(diene, old_diene_position, new_diene_position, ignore=[[diene_num1, diene_num2], [diene_num3, diene_num4]])
            if len(is_error) == 0:
                is_error = check_special_bond_ZE(diene, old_diene_position, new_diene_position, atom_ids = [[diene_num2, diene_num1], [diene_num3, diene_num4]])
                if len(is_error) == 0:
                    is_error = check_double_bond_ZE(dieno, old_dieno_position, new_dieno_position)
                    if len(is_error) == 0:
                        # is_right = up_down_check(position, diene_num1, diene_num4, dieno_num1, start, diene)
                        # if is_right:
                        new_conformer = Chem.rdchem.Conformer(new_mol.GetNumAtoms())
                        new_conformer.SetId(new_conf_id)
                        position = conformer.GetPositions()
                        for i, atom in enumerate(new_mol.GetAtoms()):
                            new_conformer.SetAtomPosition(i, position[i])
                        new_mol.AddConformer(new_conformer)
                        break
        except:
            continue
    # new_conformer = Chem.rdchem.Conformer(new_mol.GetNumAtoms())
    # new_conformer.SetId(new_conf_id)
    # position = conformer.GetPositions()
    # for i, atom in enumerate(new_mol.GetAtoms()):
    #     new_conformer.SetAtomPosition(i, position[i])
    # new_mol.AddConformer(new_conformer)
    # if len(new_mol.GetConformers()) == 0:
    #     return None, 0
        
    # ts_eng = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
    #     new_mol).CalcEnergy()
    # diene_eng = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
    #     diene).CalcEnergy()
    # dieno_eng = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
    #     dieno).CalcEnergy()
    # delta_eng = ts_eng - diene_eng - dieno_eng
    return new_mol, 0


def react(diene_mol, dieno_mol, distence=1.5, select_diene=0, FFopt=True):
    """Construct and rank Diels–Alder products for a diene–ene pair.

    Args:
        diene_mol (mol): 
        dieno_mol (mol): ene
        distence (float, optional): Initial separation. Defaults to 1.5.
        select_diene (int, optional): Diene-site selection mode. Defaults to no filtering (0).

    Returns:
        _type_: _description_
    """    
    return_mols = []
    start = diene_mol.GetNumAtoms()
    diene_group = change_position(
        diene_mol, prop="diene", addHs=True, select_diene=select_diene)
    dieno_group = change_position(dieno_mol, prop="dieno")
    for [diene, diene_list] in diene_group:
        for [dieno, dieno_list] in dieno_group:
            y_dieno_list = [dieno_list[1], dieno_list[0]]
            dieno_bond = dieno.GetBondBetweenAtoms(
                dieno_list[0], dieno_list[1])
            # Combination in 4 ways:
            # dieno_bond.SetBondType(Chem.BondType.SINGLE)
            dieno = copy.deepcopy(dieno)
            dieno_0 = move_mol(dieno, np.array([0, 0, distence]))
            dieno_x = move_mol(rot_mol(dieno, np.array(
                [1, 0, 0]), sin=0, cos=-1), np.array([0, 0, distence]))
            dieno_y = move_mol(rot_mol(dieno, np.array(
                [0, 1, 0]), sin=0, cos=-1), np.array([0, 0, distence]))
            dieno_xy = move_mol(rot_mol(rot_mol(dieno, np.array(
                [1, 0, 0]), sin=0, cos=-1), np.array([0, 1, 0]), sin=0, cos=-1), np.array([0, 0, distence]))
            title_0 = list([diene_list[0], diene_list[-1]]) + list(dieno_list) + [start] + [1]
            title_x = list([diene_list[0], diene_list[-1]]) + list(dieno_list) + [start] + [0]
            title_y = list([diene_list[0], diene_list[-1]]) + list(y_dieno_list) + [start] + [1]
            title_xy = list([diene_list[0], diene_list[-1]]) + list(y_dieno_list) + [start] + [0]

            diene_rot_x, diene_rot_y, diene_rot_xy = diene_ene_rotation_need(diene, "diene", diene_list)
            dieno_rot_x, dieno_rot_y, dieno_rot_xy = diene_ene_rotation_need(dieno, "dieno", dieno_list)
            rot_x = diene_rot_x * dieno_rot_x
            rot_y = diene_rot_y * dieno_rot_y
            rot_xy = diene_rot_xy * dieno_rot_xy

            rwcomb, deltaE = comb_mol(diene, dieno_0, diene_list, dieno_list, FF_opt=FFopt)
            if rwcomb != None:
                return_mols.append([deltaE, rwcomb, title_0])
            
            if rot_x:
                rwcomb_x, deltaE_x = comb_mol(
                    diene, dieno_x, diene_list, dieno_list, FF_opt=FFopt)
                if rwcomb_x != None:
                    return_mols.append([deltaE_x, rwcomb_x, title_x])

            if rot_y:
                rwcomb_y, deltaE_y = comb_mol(
                    diene, dieno_y, diene_list, y_dieno_list, FF_opt=FFopt)
                if rwcomb_y != None:
                    return_mols.append([deltaE_y, rwcomb_y, title_y])

            if rot_xy:
                rwcomb_xy, deltaE_xy = comb_mol(
                    diene, dieno_xy, diene_list, y_dieno_list, FF_opt=FFopt)
                if rwcomb_xy != None:
                    return_mols.append([deltaE_xy, rwcomb_xy, title_xy])

    # return_mols.sort()

    return return_mols

def eng_to_om(log_file:logfile_process.Logfile, diene_mol, new_dir="om", assert_title=None, write_gjf=True, distance=2.1, 
method='opt=modredundant freq b3lyp/6-31g(d) em=gd3bj', target_cos = -0.9396926, difreeze=True):
    """Read an optimized product log, extend the forming bonds, and prepare a constrained optimization.

    Args:
        log_file (Logfile): Parsed Gaussian log file.
    """    
    new_name = os.path.split(log_file.file_dir)[-1].split(".")[0] + ".gjf"
    newfile = new_dir + "/" + new_name 
    title = log_file.title
    charge = log_file.charge
    symbol_list = log_file.symbol_list
    position = log_file.running_positions[-1]
    # title, charge, symbol_list, position = read_log(file_name, allow_unreal_freq=1)
    if assert_title:
        title = assert_title
    diene_num1, diene_num2, dieno_num1, dieno_num2, start = [
        int(each) for each in title][:5]
    diene_point1 = position[diene_num1]
    diene_point2 = position[diene_num2]
    dieno_point1 = position[dieno_num1 + start]
    dieno_point2 = position[dieno_num2 + start]

    # Align the reactive planes.
    new_position = trfm_rot(diene_point1, diene_point2, (dieno_point1 + dieno_point2)/2, position)
    diene_center_point = (
        new_position[diene_num1] + new_position[diene_num2])/2
    dieno_center_point = (
        new_position[dieno_num1 + start] + new_position[dieno_num2 + start])/2
    y_distance = distance - np.sqrt((diene_center_point - dieno_center_point)
                                @ (diene_center_point - dieno_center_point).T)  # Target separation distance.
    diene_position = new_position[:start]
    dieno_position = new_position[start:]
    # Extend the forming bonds.
    move_matrix = move(np.array([0, y_distance, 0]))
    dieno_position = (move_matrix @ dieno_position.T).T
    # Rotate into a common plane.
    diene_array = diene_position[diene_num1] - diene_position[diene_num2]
    dieno_array = dieno_position[dieno_num1] - dieno_position[dieno_num2]
    law_array = np.cross(diene_array[:3], dieno_array[:3])
    cos = (diene_array @ dieno_array)/(np.sqrt(diene_array @ diene_array) * np.sqrt(dieno_array @ dieno_array))
    sin = np.sqrt(1 - cos ** 2)
    rot_matrix = rotation(law_array, sin, cos)
    diene_position = (rot_matrix @ diene_position.T).T
    new_position = np.append(diene_position, dieno_position, axis=0)
    new_position = np.array([each[:3] for each in new_position])
    # Reposition the coordinates.
    diene_point1 = new_position[diene_num1]
    diene_point2 = new_position[diene_num2]
    dieno_point1 = new_position[dieno_num1 + start]
    dieno_point2 = new_position[dieno_num2 + start]
    diene_atom_lists = diene_atom_Idx(diene_mol, select_diene=0)
    for each in diene_atom_lists:
        if each[0] == diene_num1 and each[-1] == diene_num2:
            atomb_id, atomc_id = each[1:3]
            diene_atom_list = each
            break
    function_groups = set()
    function_groups.update(find_sustation_group(diene_mol, diene_num1, [atomb_id, atomc_id]))
    function_groups.update(find_sustation_group(diene_mol, diene_num2, [atomb_id, atomc_id]))

    diene_point3 = new_position[atomb_id]
    # print(diene_point3)
    new2_position = trfm_rot(diene_point1, diene_point2, diene_point3, new_position)
    # new2_position = np.array([each[:3] for each in new2_position])
    # Calculate the angle.
    diene_point1 = new2_position[diene_num1][:3]
    diene_point2 = new2_position[diene_num2][:3]
    dieno_point1 = new2_position[dieno_num1 + start][:3]
    dieno_point2 = new2_position[dieno_num2 + start][:3]
    diene_point3 = new2_position[atomb_id][:3]
    
    now_cos = Tool.get_torsion(diene_point3, diene_point1, diene_point2, dieno_point2)
    if target_cos != None:
        if now_cos < target_cos or dieno_point1[2] < 0:
            diene_position = new2_position[:start]
            dieno_position = new2_position[start:]
            if dieno_point1[2] > 0:
                cos = (target_cos * now_cos + np.sqrt(1-target_cos**2) * np.sqrt(1-now_cos ** 2))
            else:
                cos = (target_cos * now_cos - np.sqrt(1-target_cos**2) * np.sqrt(1-now_cos ** 2))
            sin = -np.sqrt(1-cos ** 2)
            rot_matrix = rotation(np.array([1, 0, 0]), sin, cos)
            dieno_position = (rot_matrix @ dieno_position.T).T
            new2_position = np.append(diene_position, dieno_position, axis=0)

            # Move the substituent.
            if cos < 0.5:
                cos = 0.5
                sin = -np.sqrt(1-cos ** 2)
                rot_matrix = rotation(np.array([1, 0, 0]), sin, cos)
            new2_position = np.array(new2_position)
            sustation_position = new2_position[list(function_groups)]
            sustation_position = (rot_matrix @ sustation_position.T).T
            for sustation_id, position_id in enumerate(list(function_groups)):
                new2_position[position_id] = sustation_position[sustation_id]


    title = " ".join(str(each) for each in title)
    # savechk = new_name.split(".")[0]
    if write_gjf:
        if difreeze:
            format_change.block_to_gjf(symbol_list, new2_position, newfile, charge, title,
                    method=method,
                    freeze=[[diene_num1 + 1, dieno_num1 + start + 1], [diene_num2 + 1, dieno_num2 + start + 1]], 
                    difreeze=[[atomb_id + 1, diene_num1 + 1, diene_num2 + 1, dieno_num1 + start + 1], [atomc_id + 1, diene_num2 + 1, diene_num1 + 1, dieno_num2 + start + 1]])
        else:
            format_change.block_to_gjf(symbol_list, new2_position, newfile, charge, title,
                    method=method,
                    freeze=[[diene_num1 + 1, dieno_num1 + start + 1], [diene_num2 + 1, dieno_num2 + start + 1]])
    return title, symbol_list, new2_position, charge, diene_atom_list

def om_to_om2(log_file, 
            diene_mol, 
            new_dir="om", 
            assert_title=None, 
            write_gjf=True, 
            distance=1.9, 
            method='opt=modredundant freq b3lyp/6-31g(d) em=gd3bj',
            target_cos = -np.sqrt(3)/2, 
            freeze = True, 
            difreeze = True):
    if type(log_file) in [list, tuple]:
        new_name, title, charge, symbol_list, position = log_file
    else:
        new_name = os.path.split(log_file.file_dir)[-1].split(".")[0] + ".gjf"
        title = log_file.title
        charge = log_file.charge
        symbol_list = log_file.symbol_list
        position = log_file.running_positions[-1]
    newfile = new_dir + "/" + new_name
    # title, charge, symbol_list, position = read_log(file_name, allow_unreal_freq=1)
    if assert_title:
        title = assert_title
    diene_num1, diene_num2, dieno_num1, dieno_num2, start = [
        int(each) for each in title][:5]

    # Reposition the coordinates.
    position = np.array(position) + 0.0001
    diene_point1 = position[diene_num1]
    diene_point2 = position[diene_num2]
    dieno_point1 = position[dieno_num1 + start]
    dieno_point2 = position[dieno_num2 + start]
    diene_atom_lists = diene_atom_Idx(diene_mol, select_diene=0)
    for each in diene_atom_lists:
        if each[0] == diene_num1 and each[-1] == diene_num2:
            atomb_id, atomc_id = each[1:3]
            break
    function_groups = set()
    function_groups.update(find_sustation_group(diene_mol, diene_num1, [atomb_id, atomc_id]))
    function_groups.update(find_sustation_group(diene_mol, diene_num2, [atomb_id, atomc_id]))


    diene_point3 = position[atomb_id]
    # print(diene_point3)
    new2_position = trfm_rot(diene_point1, diene_point2, diene_point3, position)
    # new2_position = np.array([each[:3] for each in new2_position])
    # Calculate the angle.
    diene_point1 = new2_position[diene_num1][:3]
    diene_point2 = new2_position[diene_num2][:3]
    dieno_point1 = new2_position[dieno_num1 + start][:3]
    dieno_point2 = new2_position[dieno_num2 + start][:3]
    diene_point3 = new2_position[atomb_id][:3]
    
    now_cos = Tool.get_torsion(diene_point3, diene_point1, diene_point2, dieno_point2)
    # target_cos = -np.sqrt(2)/2
    if target_cos != None:
        if now_cos < target_cos:
            diene_position = new2_position[:start]
            dieno_position = new2_position[start:]
            cos = (target_cos * now_cos + np.sqrt(1-target_cos**2) * np.sqrt(1-now_cos ** 2))
            sin = -np.sqrt(1-cos ** 2)
            rot_matrix = rotation(np.array([1, 0, 0]), sin, cos)
            dieno_position = (rot_matrix @ dieno_position.T).T
            new2_position = np.append(diene_position, dieno_position, axis=0)

    title = " ".join(str(each) for each in title)
    # savechk = new_name.split(".")[0]
    if freeze:
        freeze = [[diene_num1 + 1, dieno_num1 + start + 1], [diene_num2 + 1, dieno_num2 + start + 1]]
    else:
        freeze = []
    if difreeze: 
        difreeze=[[atomb_id + 1, diene_num1 + 1, diene_num2 + 1, dieno_num1 + start + 1], [atomc_id + 1, diene_num2 + 1, diene_num1 + 1, dieno_num2 + start + 1]]
    else:
        difreeze=[]
    if write_gjf:
        format_change.block_to_gjf(symbol_list, new2_position, newfile, charge, title,
                method=method,
                freeze=freeze, 
                difreeze=difreeze)
    return title, symbol_list, new2_position, charge, diene_atom_lists



def om_to_ts(log_file:logfile_process.Logfile, write_gjf=True, new_dir="ts",
    method="opt=(calcfc,ts,noeigen) freq b3lyp/6-31g* em=gd3bj"):
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    new_name = os.path.split(log_file.file_dir)[-1].split(".")[0] + ".gjf"
    newfile = new_dir + "/" + new_name 
    title = log_file.title
    charge = log_file.charge
    symbol_list = log_file.symbol_list
    try:
        position = log_file.running_positions[-1]
    except:
        position = log_file.first_atom_position
    title = " ".join(str(each) for each in title)
    if write_gjf:
        format_change.block_to_gjf(symbol_list, position, newfile, charge, title,
                    method=method,
                    freeze=[])
    
    return title, symbol_list, position, charge


def ts_to_irc(log_file:logfile_process.Logfile, new_dir, methods=None):
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    new_names = [os.path.split(log_file.file_dir)[-1].split(".")[0] + "forward.gjf",
    os.path.split(log_file.file_dir)[-1].split(".")[0] + "reverse.gjf"]
    if methods == None:
        methods = ['irc=(calcfc,stepsize=30,maxpoints=20,forward, lqa) b3lyp/6-31g(d)', 
        'irc=(calcfc,stepsize=30,maxpoints=20,reverse, lqa) b3lyp/6-31g(d)']

    title = log_file.title
    title = " ".join(str(each) for each in title)
    charge = log_file.charge
    symbol_list = log_file.symbol_list
    position = log_file.running_positions[-1]
    for new_name, method in zip(new_names, methods):
        newfile = new_dir + "/" + new_name 
        format_change.block_to_gjf(symbol_list, position, newfile, charge, title,
                    method=method,
                    freeze=[])
    
    return title, symbol_list, position, charge

def smiles2mol(smiles, conf_num=20):
    """Convert SMILES to a molecule, add hydrogens, and generate 3D conformers.

    Args:
        smiles (str): _description_

    Returns:
        mol: _description_
    """    
    mol = Chem.MolFromSmiles(smiles)
    if mol == None:
        print(smiles, "could not be parsed")
        return None
    Hmol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(Hmol, numConfs=conf_num, maxAttempts=100, )
    try:
        AllChem.MMFFOptimizeMolecule(Hmol)
    except:
        pass
    
    return Hmol

def add_conformer(mol, conf_num=50):
    for _ in range(conf_num):
        b = copy.deepcopy(mol)
        Chem.AllChem.MMFFOptimizeMolecule(b)
        mol.AddConformer(b.GetConformer(0), assignId=True)
    return mol

def find_sustation_group(mol, mother_atom:int, ignore_atoms = []):
    """Find all atoms in the substituent attached to mother_atom, excluding ignore_atoms.

    Args:
        mol (_type_): _description_
        mother_atom (int): _description_
        ignore_atoms (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """    
    all_atoms = set()
    neighbor = [each.GetIdx() for each in mol.GetAtomWithIdx(mother_atom).GetNeighbors() if each.GetIdx() not in ignore_atoms]
    all_atoms.update(neighbor)
    for atom in neighbor:
        new_ignore_atoms = ignore_atoms + [atom]
        all_atoms.update(find_sustation_group(mol, atom, new_ignore_atoms))
    return all_atoms

# def read_reactant(csvfile, index_lists=None):
#     """Read a CSV containing diene/ene indices, SMILES, and energies.

#     Args:
#         csvfile (_type_): _description_
#         index_lists (_type_, optional): _description_. Defaults to None.

#     Returns:
#         _type_: _description_
#     """    
#     file = pd.read_csv(csvfile, index_col="Index").to_numpy()
#     if index_lists ==None:
#         smiles, energy = file[:, 0], file[:, -1]
#     else:
#         smiles, energy = file[index_lists, 0], file[index_lists, -1]
#     return smiles, energy


# Fallback for difficult conformers: fragment the molecule.
# from functools import reduce
# def is_sp3(atom):
#     """Check whether an atom is sp3-hybridized.

#     Args:
#         atom (rdkit.Atom): 

#     Returns:
#         bool: 
#     """    
#     num_bonds = atom.GetTotalDegree()
#     hybridization = atom.GetHybridization()
#     return num_bonds == 4 and hybridization == Chem.rdchem.HybridizationType.SP3
    
# def find_common(list_of_set):
#     """Return the intersection of multiple sets.

#     Args:
#         list_of_set (_type_): _description_

#     Returns:
#         _type_: _description_
#     """    
#     return set(reduce(np.intersect1d, np.array([list(each) for each in list_of_set])))
# def find_all_in(list_of_set):
#     # Return the union of multiple sets.
#     return_set = []
#     for each in [list(return_atom_num_set) for return_atom_num_set in list_of_set]:
#         return_set += each
#     return_set = set(return_set)
#     return return_set
    

# def find_neighbor_atom_number(mol, centers, times=3, require_sp3=True,consider_ring=True, exclude_atoms = []):
#     """Find atom indices within a specified number of bonds from the center atoms.

#     Args:
#         mol (_type_): _description_
#         centers (_type_): _description_
#         times (int, optional): _description_. Defaults to 3.
#         require_sp3 (bool, optional): _description_. Defaults to True.
#         consider_ring (bool, optional): _description_. Defaults to True.

#     Returns:
#         _type_: _description_
#     """    
#     step = 0
#     if len(centers) == 1:
#         return_atom_num_set = set(centers[0])
#         temp_set = set(centers[0])
#     else:
#         return_atom_num_sets = []
#         temp_sets = [] 
#         for center_id, center in enumerate(centers):
#             return_atom_num_set = set()
#             temp_set = set()
#             for each in center:
#                 return_atom_num_set.add(each)
#                 temp_set.add(each)
#             return_atom_num_sets.append(return_atom_num_set)
#             temp_sets.append(temp_set)

#         while (True):
#             step += 1
#             for center_id, center in enumerate(centers):
#                 new_num_set = set()
#                 for eachatomnum in temp_sets[center_id]:
#                     Atom = mol.GetAtomWithIdx(eachatomnum)
#                     neighbors = set([atom.GetIdx() for atom in Atom.GetNeighbors() if atom.GetIdx() not in return_atom_num_sets[center_id] and atom.GetIdx() not in exclude_atoms])
#                     for eachring in Chem.GetSymmSSSR(mol):
#                         if len(neighbors & set(eachring)) > 0:
#                             neighbors = neighbors | set(eachring)
#                     new_num_set = new_num_set|neighbors
#                     return_atom_num_sets[center_id] = return_atom_num_sets[center_id] | new_num_set
#                 temp_sets[center_id] = new_num_set
#             common_atom = find_common(return_atom_num_sets)
#             if len(common_atom):
#                 return_atom_num_set = find_all_in(return_atom_num_sets)
#                 temp_set = find_all_in(temp_sets)
#                 break

#     while (True):
#         if step >= times:
#             if sum([is_sp3(mol.GetAtomWithIdx(atomid)) for atomid in temp_set]) == len(temp_set) or not require_sp3:
#                 break
#         step += 1
#         new_num_set = set()
#         for eachatomnum in temp_set:
#             Atom = mol.GetAtomWithIdx(eachatomnum)
#             neighbors = set([atom.GetIdx() for atom in Atom.GetNeighbors() if atom.GetIdx() not in return_atom_num_set and atom.GetIdx() not in exclude_atoms])
#             if consider_ring and require_sp3:
#                 for eachring in Chem.GetSymmSSSR(mol):
#                     if len(neighbors & set(eachring)) > 0:
#                         neighbors = neighbors | set(eachring)
#             new_num_set = new_num_set|neighbors
#         return_atom_num_set= return_atom_num_set | new_num_set
#         temp_set = new_num_set
    
#     if consider_ring and not require_sp3:
#         for eachatomnum in centers[0]:
#             for eachring in Chem.GetSymmSSSR(mol):
#                 if eachatomnum in eachring:
#                     return_atom_num_set = return_atom_num_set | set(eachring)

                    

#     #     for eachring in Chem.GetSymmSSSR(mol):
#     #         if len(return_atom_num_set & set(eachring)) > 0:
#     #             return_atom_num_set = return_atom_num_set | set(eachring)
#     return return_atom_num_set

def is_tran_cycloene(smiles, assert_bond_type=True, return_ring=False):
    mol = Chem.MolFromSmiles(smiles)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetSymbol() == "C" and atom2.GetSymbol() == "C":
                atom1_id = atom1.GetIdx()
                atom2_id = atom2.GetIdx()
                for ring in Chem.GetSymmSSSR(mol):
                    if atom1_id in list(ring) and atom2_id in list(ring):
                        if bond.GetStereo() == Chem.BondStereo.STEREOE or not assert_bond_type:
                            print("is_tran_cycloene, %s" % smiles)
                            if return_ring:
                                return (True, bond.GetIdx(), list(ring))
                            else:
                                return (True, bond.GetIdx())
    if return_ring:
        return (False, 0, 0)
    else:
        return (False, 0)

def to_trans_cycloene(smiles="C1=C/CCCCCC/1", assert_bond_type=True, test_mod=False, conf_num=50):
    def checkZE(mol, conf_id=0):
        atom1 = [atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 101][0]
        atom2 = [atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 102][0]
        another_atom1 = [atom for atom in atom1.GetNeighbors() if atom.GetAtomMapNum() == 50][0]
        another_atom2 = [atom for atom in atom2.GetNeighbors() if atom.GetAtomMapNum() == 50][0]
        position = mol.GetConformer(conf_id).GetPositions()
        cos = Tool.get_torsion(position[another_atom1.GetIdx()], position[atom1.GetIdx()], position[atom2.GetIdx()], position[another_atom2.GetIdx()])
        return cos
    
    
    tf, id, ring = is_tran_cycloene(smiles, assert_bond_type, return_ring=True)
    if tf:
        mol = Chem.MolFromSmiles(smiles)
        if "H" not in smiles:
            mol = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        # AllChem.EmbedMultipleConfs(mol, numConfs=conf_num, maxAttempts=100, )
        bond = mol.GetBondWithIdx(id)
        atom1_id, atom2_id = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rwmol = Chem.RWMol(mol)
        atom_id = rwmol.AddAtom(Chem.Atom("C"), )
        rbond = rwmol.GetBondWithIdx(id)
        rbond.SetBondType(Chem.BondType.SINGLE)
        rwmol.AddBond(atom_id, atom1_id, Chem.BondType.SINGLE)
        rwmol.AddBond(atom_id, atom2_id, Chem.BondType.SINGLE)
        mol = rwmol.GetMol()
        atom1 = mol.GetAtomWithIdx(atom1_id)
        atom1.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        atom2 = mol.GetAtomWithIdx(atom2_id)
        atom2.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        # Sanitize the molecule after bond edits.
        AllChem.SanitizeMol(mol)
        for each in ring:
            mol.GetAtomWithIdx(each).SetAtomMapNum(50)
        mol.GetAtomWithIdx(atom1_id).SetAtomMapNum(101)
        mol.GetAtomWithIdx(atom2_id).SetAtomMapNum(102)
        mol.GetAtomWithIdx(atom_id).SetAtomMapNum(100)
        smiles = Chem.MolToSmiles(mol)
        amol = smiles2mol(smiles, conf_num)

        # Invert the stereochemistry if the alkene geometry is incorrect.
        if checkZE(amol, 0) > 0:
            atom2 = mol.GetAtomWithIdx(atom2_id)
            atom2.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            AllChem.SanitizeMol(mol)
            mol.GetAtomWithIdx(atom_id).SetAtomMapNum(100)
            smiles = Chem.MolToSmiles(mol)
            amol = smiles2mol(smiles, conf_num)
        # Keep conformers with the required stereochemistry.
        new_mol = copy.deepcopy(amol)
        Chem.AllChem.EmbedMultipleConfs(new_mol, 0)
        new_conf_id = 0
        for conf_id, conformer in enumerate(amol.GetConformers()):
            if checkZE(amol, conf_id=conf_id) < 0:
                # select_mol
                new_conformer = Chem.rdchem.Conformer(amol.GetNumAtoms())
                new_conformer.SetId(new_conf_id)
                position = conformer.GetPositions()
                for i, atom in enumerate(amol.GetAtoms()):
                    new_conformer.SetAtomPosition(i, position[i])
                new_mol.AddConformer(new_conformer)
                new_conf_id += 1
        # Remove temporary atoms.
        if test_mod:
            return new_mol
        new_atom = [atom for atom in new_mol.GetAtoms() if atom.GetAtomMapNum() == 100][0]
        new_atom.SetAtomMapNum(10)
        for atom in new_atom.GetNeighbors():
            if atom.GetSymbol() == "H":
                atom.SetAtomMapNum(10)
        rwmol = Chem.RWMol(new_mol)
        atom_id = 0
        while(atom_id < rwmol.GetNumAtoms()):
            if rwmol.GetAtomWithIdx(atom_id).GetAtomMapNum() == 10:
                rwmol.RemoveAtom(atom_id)
            else:
                atom_id += 1
        start_atom = [atom for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == 101][0]
        end_atom = [atom for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == 102][0]
        bond_id = rwmol.GetBondBetweenAtoms(start_atom.GetIdx(), end_atom.GetIdx()).GetIdx()
        rbond = rwmol.GetBondWithIdx(bond_id)
        rbond.SetBondType(Chem.BondType.DOUBLE)
        bmol = rwmol.GetMol()
        for atom in bmol.GetAtoms():
            atom.SetAtomMapNum(0)
        # bmol = copy.deepcopy(bmol)
        # AllChem.EmbedMolecule(bmol)
        return bmol 
    return smiles2mol(smiles, conf_num)


def is_same_atom(mol, atoma, atomb, ignore_atoms=[]):
    if atoma.GetSymbol() != atomb.GetSymbol():
        return False 
    if atoma.GetFormalCharge() != atomb.GetFormalCharge():
        return False
    neighbor1 = [atom for atom in atoma.GetNeighbors() if atom.GetIdx() not in ignore_atoms]
    neighbor2 = [atom for atom in atomb.GetNeighbors() if atom.GetIdx() not in ignore_atoms]
    if len(neighbor1) != len(neighbor2):
        return False
    atom_symbol_list1 = [atom.GetSymbol() for atom in neighbor1]
    atom_symbol_list2 = [atom.GetSymbol() for atom in neighbor2]
    atom_symbol_list1.sort()
    atom_symbol_list2.sort()
    if atom_symbol_list1 != atom_symbol_list2:
        return False
    bond_a = [str(each.GetBondType()) for each in [mol.GetBondBetweenAtoms(atoma.GetIdx(), each.GetIdx()) for each in neighbor1]]
    bond_b = [str(each.GetBondType()) for each in [mol.GetBondBetweenAtoms(atomb.GetIdx(), each.GetIdx()) for each in neighbor2]]
    bond_a.sort()
    bond_b.sort()
    if bond_a != bond_b:
        return False
    bond_a = [each for each in [mol.GetBondBetweenAtoms(atoma.GetIdx(), each.GetIdx()) for each in neighbor1] if each.GetBondType() == Chem.BondType.DOUBLE]
    bond_b = [each for each in [mol.GetBondBetweenAtoms(atomb.GetIdx(), each.GetIdx()) for each in neighbor2] if each.GetBondType() == Chem.BondType.DOUBLE]
    if len(bond_a) == 0:
        return True
    else:
        bond_a = bond_a[0].GetStereo()
        bond_b = bond_b[0].GetStereo()
        if bond_a != bond_b:
            return False
    return True

def search_next_atom(mol, parent_atom1, parent_atom2, pre_atoms=set()):
    if not is_same_atom(mol, parent_atom1, parent_atom2, ignore_atoms = pre_atoms):
        return False
    neighbor1 = [atom for atom in parent_atom1.GetNeighbors() if atom.GetIdx() not in pre_atoms]
    neighbor2 = [atom for atom in parent_atom2.GetNeighbors() if atom.GetIdx() not in pre_atoms]
    return_value = True
    for sub_neighbor1 in neighbor1:
        is_same = False
        pair_neighbor2 = neighbor2
        for sub_neighbor2 in pair_neighbor2:
            next_pre_atoms = set(list(pre_atoms) + [sub_neighbor1.GetIdx(), sub_neighbor2.GetIdx()])
            if search_next_atom(mol, sub_neighbor1, sub_neighbor2, next_pre_atoms):
                is_same = True
                break
        if is_same:
            pair_neighbor2.remove(sub_neighbor2)
        else:
            return False
    return True

def diene_ene_rotation_need(mol, mol_type, ene_atoms):
    assert mol_type in ["diene", "dieno"]
    need_x_rot, need_y_rot, need_xy_rot = 1, 1, 1
    atom1 = mol.GetAtomWithIdx(ene_atoms[0])
    atom2 = mol.GetAtomWithIdx(ene_atoms[-1])
    if atom1.GetSymbol() != atom2.GetSymbol():
        return 1,1,1
    if mol_type == "diene":
        atom3 = mol.GetAtomWithIdx(ene_atoms[1])
        atom4 = mol.GetAtomWithIdx(ene_atoms[-2])
        if atom3.GetSymbol() != atom4.GetSymbol():
            return 1,1,1
        if not search_next_atom(mol,atom3, atom4, ene_atoms):
            return 1,1,1
    neighbor1 = [atom.GetIdx() for atom in atom1.GetNeighbors() if atom.GetIdx() not in ene_atoms]
    neighbor2 = [atom.GetIdx() for atom in atom2.GetNeighbors() if atom.GetIdx() not in ene_atoms]
    pre_atoms = set([ene_atoms[0], ene_atoms[-1]] + neighbor1 + neighbor2)
    if len(neighbor1) == 1 and len(neighbor2) == 1 :
        # is yne
        need_x_rot = 0
        neighbor_atom1 = mol.GetAtomWithIdx(neighbor1[0])
        neighbor_atom2 = mol.GetAtomWithIdx(neighbor2[0])
        if search_next_atom(mol, neighbor_atom1, neighbor_atom2, pre_atoms):
            need_y_rot = 0
    elif len(neighbor1) == 2 and len(neighbor2) == 2:
        atoma, atomb, atomc, atomd = [mol.GetAtomWithIdx(each) for each in neighbor1 + neighbor2]
        if search_next_atom(mol, atoma, atomb, pre_atoms) and search_next_atom(mol, atomc, atomd, pre_atoms):
            need_x_rot = 0
            if search_next_atom(mol, atoma, atomc, pre_atoms):
                need_y_rot = 0
        else:
            conformer = mol.GetConformers()[0]
            position = conformer.GetPositions()
            if mol_type == "dieno":
                A, B, C, D = [np.array(position[each]) for each in [atoma.GetIdx(), ene_atoms[0], ene_atoms[-1], atomc.GetIdx()]]
                cos0 = Tool.get_torsion(A, B, C, D)
            else:
                A, B, C, D = [np.array(position[each]) for each in [atoma.GetIdx(), ene_atoms[0], ene_atoms[1],  ene_atoms[2]]]
                cos0 = Tool.get_torsion(A, B, C, D)
                A, B, C, D = [np.array(position[each]) for each in [atomc.GetIdx(), ene_atoms[-1], ene_atoms[-2],  ene_atoms[-3]]]
                cos1 = Tool.get_torsion(A, B, C, D)
                cos0 *= cos1
            if search_next_atom(mol, atoma, atomc, pre_atoms) and search_next_atom(mol, atomb, atomd, pre_atoms):
                if cos0 > 0:
                    need_y_rot = 0
                else:
                    need_x_rot = 0
            elif search_next_atom(mol, atoma, atomd, pre_atoms) and search_next_atom(mol, atomb, atomc, pre_atoms):
                if cos0 < 0:
                    need_y_rot = 0
                else:
                    need_x_rot = 0
    if mol_type == "diene":
        return 1, need_y_rot, 1 * need_y_rot
    else:
        return need_x_rot, need_y_rot, need_x_rot * need_y_rot

def check_stereo(mol_file, atom_lists, position):
    origin_mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    new_mol = copy.deepcopy(origin_mol)
    new_mol = xtb_process.xtb_to_mol(new_mol, [atom_lists], [position], 1)
    AssignStereochemistryFrom3D(origin_mol)
    AssignStereochemistryFrom3D(new_mol)
    for atoma, atomb in zip(origin_mol.GetAtoms(), new_mol.GetAtoms()):
        if atoma.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        if atomb.GetChiralTag() != atoma.GetChiralTag():
            return False
    return True
