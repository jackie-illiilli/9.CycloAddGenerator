# Common utility functions.
from inspect import BoundArguments
import math, os, shutil
import numpy as np
import pandas as pd
from rdkit import Chem

def find_first_line(fileline, find_str, find_type="start"):
    assert find_type in ['start', 'end', 'all', 'in']
    if find_type == "start":
        line_function = lambda x, y: x.startswith(y)
    elif find_type == 'end':
        line_function = lambda x, y: x.endswith(y)
    elif find_type == 'all':
        line_function = lambda x, y: x == y
    else:
        line_function = lambda x, y: y in x

    for idx, line in enumerate(fileline):
        if line_function(line, find_str):
            return idx, line
    return None, None

def remove_same(lists):
    """Remove duplicates from a list of lists.

    Args:
        lists (list): A list of lists.

    Returns:
        list: A list of lists without duplicates.
    """
    seen = set()
    return_list = []
    for each in lists:
        # Convert each list to a tuple so it can be hashed
        t = tuple(each)
        # Check if the tuple is already in the set
        if t not in seen:
            # Add it to the set and the return list
            seen.add(t)
            return_list.append(each)
    return return_list

def save_load(file_name, smiles_lists = None):
    if smiles_lists != None and len(smiles_lists) != 0:
        with open(file_name, "wt") as f:
            for each in smiles_lists:
                f.write(each + "\n")
        return None
    else:
        smiles_lists = []
        with open(file_name, "rt") as f:
            for eachline in f.readlines():
                smiles_lists.append(eachline.strip("\n"))
        print(len(smiles_lists))
        smiles_lists_ = smiles_lists
        smiles_lists = []
        return smiles_lists_

def clean_nan(input_list):
    return np.nan_to_num(input_list, nan=0)

def get_array_cos(array1, array2):
    return array1 @ array2.T / (np.sqrt(array1 @ array1.T)
                              * np.sqrt(array2 @ array2.T))

def get_atoms_distance(atom_positionA, atom_positionB):
    """[summary]

    Args:
        atom_positionA (array): [description]
        atom_positionA (array): [description]

    Returns:
        array: distance
    """
    return np.sqrt(sum((atom_positionA - atom_positionB) ** 2))

def get_bond_angle_deg(atom_positionA, atom_positionB, atom_positionC):
    conf = Chem.rdchem.Conformer(3)
    all_positions = [atom_positionA, atom_positionB, atom_positionC]
    for i in range(3):
        conf.SetAtomPosition(i, all_positions[i][:3])
    bond_angle = Chem.rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
    return bond_angle

def get_torsion(A, B, C, D):
    """Calculate the cosine of the A-B-C-D torsion angle.

    Args:
        A (array): points
        B (array): 
        C (array): 
        D (array): 

    Returns:
        cos: _description_
    """    
    AB_AC = np.cross((B - A), (C - A))
    DB_DC = np.cross((B - D), (C - D))
    cos0 = get_array_cos(AB_AC, DB_DC)

    return cos0
def get_torsion_(A, B, C, D):
    conf = Chem.rdchem.Conformer(4)
    all_positions = [A, B, C, D]
    for i in range(4):
        conf.SetAtomPosition(i, all_positions[i][:3])
    bond_angle = Chem.rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)
    return bond_angle

def GetSpinMultiplicity(Mol, CheckMolProp = True):
    """From RDKitUtil.py
    Get spin multiplicity of a molecule. The spin multiplicity is either
    retrieved from 'SpinMultiplicity' molecule property or calculated from
    from the number of free radical electrons using Hund's rule of maximum
    multiplicity defined as 2S + 1 where S is the total electron spin. The
    total spin is 1/2 the number of free radical electrons in a molecule.

    Arguments:
        Mol (object): RDKit molecule object.
        CheckMolProp (bool): Check 'SpinMultiplicity' molecule property to
            retrieve spin multiplicity.

    Returns:
        int : Spin multiplicity.

    """
    
    Name = 'SpinMultiplicity'
    if (CheckMolProp and Mol.HasProp(Name)):
        return int(float(Mol.GetProp(Name)))

    # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
    NumRadicalElectrons = 0
    for Atom in Mol.GetAtoms():
        NumRadicalElectrons += Atom.GetNumRadicalElectrons()

    TotalElectronicSpin = NumRadicalElectrons/2
    SpinMultiplicity = 2 * TotalElectronicSpin + 1
    
    return int(SpinMultiplicity)


def stablize_smileses(smiles_list):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(each)) for each in smiles_list]

