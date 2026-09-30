"""MeS workflow functions extracted without execution from the archived notebook.
No Gaussian input generation or result collection runs on import.
"""
from IPython.display import display
from _workflow_paths import REPO_ROOT, SECONDARY_ROOT, SMILES_ROOT, MES_OUTPUT_DIR, MES_INPUT_CSV
from decimal import Decimal
from pathlib import Path
import math

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from Code import cycle_process, format_change, logfile_process, main


PROJECT_ROOT = REPO_ROOT
INPUT_CSV_CANDIDATES = [
    PROJECT_ROOT / "Data" / "Predict_DI" / "7_MAAtest_DA.csv",
    PROJECT_ROOT / "Data" / "Predict_DI" / "4p-2p" / "7_MAAtest_DA.csv",
]
INPUT_CSV = next(
    (path for path in INPUT_CSV_CANDIDATES if path.exists()),
    INPUT_CSV_CANDIDATES[0],
)

ALL_SMILES_CSV = SMILES_ROOT / "all_smiles.csv"
MOL_DIR = SMILES_ROOT / "mol"
OPT_LOG_DIR = SMILES_ROOT / "mol_dft_eng"

INPUT_CSV = MES_INPUT_CSV
OUTPUT_DIR = MES_OUTPUT_DIR
PRODUCT_METHOD = "opt freq b3lyp/6-31g* em=gd3bj"
TS_METHOD = "opt=(calcfc,ts,noeigen,maxstep=5) freq b3lyp/6-31g* em=gd3bj"
RELAXED_SCAN_METHOD = "opt=modredundant freq b3lyp/6-31g* em=gd3bj"
ENG_METHOD = "b3lyp/6-311+g(d,p) em=gd3bj scrf=(smd, solvent=water)"

PRODUCT_CS_DISTANCE = 1.85
SC_DISTANCE = 1.82
CH_DISTANCE = 1.09
TETRAHEDRAL_ANGLE_DEG = 109.4712206

SCAN_FACTOR_START = 1.00
SCAN_FACTOR_STOP = 1.50
SCAN_FACTOR_STEP = 0.02
HARTREE_TO_KCAL = 627.509474


# Original 7_MAA_calc_MeS.ipynb, zero-based cell 3.
def to_int(value):
    """Convert csv values such as 13033.0, "13033", or numpy scalars to int."""
    if pd.isna(value):
        raise ValueError("Cannot convert NaN to int")
    return int(float(value))

def parse_reactive_sites(title):
    parts = str(title).split()
    if len(parts) < 2:
        raise ValueError(f"Title does not contain two atom ids: {title!r}")
    return to_int(parts[0]), to_int(parts[1])

def load_stable_conf_table(all_smiles_csv=ALL_SMILES_CSV):
    table = pd.read_csv(all_smiles_csv)
    required = {"Index", "Stable_conf_id"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"Missing columns in {all_smiles_csv}: {sorted(missing)}")
    table = table.copy()
    table["Index_int"] = table["Index"].map(to_int)
    table["Stable_conf_id_int"] = table["Stable_conf_id"].map(to_int)
    return table.set_index("Index_int", drop=False)

def get_stable_conf_id(diene_index, stable_table):
    diene_index = to_int(diene_index)
    if diene_index not in stable_table.index:
        raise KeyError(f"Diene index {diene_index} is not in {ALL_SMILES_CSV}")
    return to_int(stable_table.loc[diene_index, "Stable_conf_id_int"])

def find_optimized_log(diene_index, stable_conf_id, opt_log_dir=OPT_LOG_DIR):
    diene_index = to_int(diene_index)
    stable_conf_id = to_int(stable_conf_id)
    candidates = [
        Path(opt_log_dir) / f"smilesid_{diene_index:05d}_{stable_conf_id:04d}.log",
        Path(opt_log_dir) / f"smilesid_{diene_index:05d}_{stable_conf_id}.log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find optimized log. Tried: " + "; ".join(str(p) for p in candidates))

def find_mol_file(diene_index, mol_dir=MOL_DIR):
    diene_index = to_int(diene_index)
    mol_path = Path(mol_dir) / f"smilesid_{diene_index:05d}.mol"
    if not mol_path.exists():
        raise FileNotFoundError(f"Cannot find mol file: {mol_path}")
    return mol_path

def final_log_position(log_file):
    if hasattr(log_file, "running_positions") and log_file.running_positions is not None and len(log_file.running_positions):
        return np.asarray(log_file.running_positions[-1], dtype=float)
    return np.asarray(log_file.first_atom_position, dtype=float)

def normalize_symbols(symbol_list):
    periodic_table = Chem.GetPeriodicTable()
    normalized = []
    for symbol in symbol_list:
        text = str(symbol)
        if text.isdigit():
            normalized.append(periodic_table.GetElementSymbol(int(text)))
        else:
            normalized.append(text)
    return normalized

def read_diene_structure(diene_index, stable_conf_id):
    log_path = find_optimized_log(diene_index, stable_conf_id)
    mol_path = find_mol_file(diene_index)

    mol = Chem.MolFromMolFile(str(mol_path), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit failed to read mol file: {mol_path}")

    log_file = logfile_process.Logfile(str(log_path))
    if not hasattr(log_file, "symbol_list") or log_file.symbol_list is None:
        raise ValueError(f"Failed to read symbols from log file: {log_path}")

    symbols = normalize_symbols(log_file.symbol_list)
    position = final_log_position(log_file)
    charge = int(log_file.charge)

    if len(symbols) != len(position):
        raise ValueError(f"Symbol/coordinate length mismatch in {log_path}")
    if mol.GetNumAtoms() != len(position):
        raise ValueError(
            f"Mol/log atom count mismatch for smilesid_{to_int(diene_index):05d}: "
            f"mol={mol.GetNumAtoms()}, log={len(position)}"
        )

    return symbols, position, charge, mol, log_path, mol_path

# Original 7_MAA_calc_MeS.ipynb, zero-based cell 4.
def unit_vector(vector, fallback=None):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm > 1.0e-12:
        return vector / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0])
    fallback = np.asarray(fallback, dtype=float)
    fallback_norm = np.linalg.norm(fallback)
    if fallback_norm <= 1.0e-12:
        raise ValueError("Fallback vector also has zero length")
    return fallback / fallback_norm

def perpendicular_basis(axis):
    """Return two unit vectors perpendicular to axis."""
    axis = unit_vector(axis)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    u = unit_vector(np.cross(axis, trial))
    v = unit_vector(np.cross(axis, u))
    return u, v

def choose_alignment_atom(mol, a, b):
    """Pick an atom connected to a or b, preferring a heavy atom."""
    atom_count = mol.GetNumAtoms()
    if not (0 <= a < atom_count and 0 <= b < atom_count):
        raise IndexError(f"Reactive atoms out of range: a={a}, b={b}, atom_count={atom_count}")

    candidates = []
    for center, other in ((a, b), (b, a)):
        for neighbor in mol.GetAtomWithIdx(center).GetNeighbors():
            if neighbor.GetIdx() != other:
                candidates.append(neighbor.GetIdx())
    if not candidates:
        for center in (a, b):
            candidates.extend(neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(center).GetNeighbors())
    if not candidates:
        raise ValueError(f"Cannot find an alignment neighbor connected to atom {a} or {b}")

    heavy = [idx for idx in candidates if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1]
    return heavy[0] if heavy else candidates[0]

def align_reactive_site(position, mol, a, b):
    alignment_atom = choose_alignment_atom(mol, a, b)
    aligned = cycle_process.trfm_rot(
        position[a], position[b], position[alignment_atom], position
    )
    return np.asarray(aligned, dtype=float)[:, :3], alignment_atom

def build_methanethiolate_fragment(
    aligned_position,
    a,
    b,
    attack_label,
    face,
    product_cs_distance=PRODUCT_CS_DISTANCE,
    sc_distance=SC_DISTANCE,
    ch_distance=CH_DISTANCE,
):
    """Return coordinates ordered as S, C, H, H, H."""
    if attack_label not in {"a", "b"}:
        raise ValueError("attack_label must be 'a' or 'b'")
    if face not in {"up", "down"}:
        raise ValueError("face must be 'up' or 'down'")

    attack_atom = a if attack_label == "a" else b
    sign = 1.0 if face == "up" else -1.0
    outward = np.array([0.0, 0.0, sign])

    sulfur = aligned_position[attack_atom] + product_cs_distance * outward
    methyl_c = sulfur + sc_distance * outward

    # The C-S direction is one tetrahedral substituent of methyl carbon.
    c_to_s = unit_vector(sulfur - methyl_c)
    u, v = perpendicular_basis(c_to_s)
    axial = math.cos(math.radians(TETRAHEDRAL_ANGLE_DEG))
    radial = math.sin(math.radians(TETRAHEDRAL_ANGLE_DEG))
    hydrogens = []
    for azimuth_deg in (0.0, 120.0, 240.0):
        phi = math.radians(azimuth_deg)
        direction = axial * c_to_s + radial * (
            math.cos(phi) * u + math.sin(phi) * v
        )
        hydrogens.append(methyl_c + ch_distance * direction)

    return np.vstack([sulfur, methyl_c, *hydrogens])

def set_conformer_positions(mol, position):
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, coord in enumerate(np.asarray(position, dtype=float)):
        conf.SetAtomPosition(
            atom_idx, Point3D(float(coord[0]), float(coord[1]), float(coord[2]))
        )
    mol.AddConformer(conf, assignId=True)
    return mol

def conformer_positions(mol):
    conf = mol.GetConformer(0)
    return np.array([
        [
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        ]
        for i in range(mol.GetNumAtoms())
    ])

def find_diene_atom_order(mol, a, b):
    """Find the a-b double-single-double path and return it in a -> b order."""
    candidates = cycle_process.diene_atom_Idx(Chem.Mol(mol), select_diene=0)
    for atom_order in candidates:
        atom_order = [int(each) for each in atom_order]
        if atom_order[0] == a and atom_order[-1] == b:
            return atom_order
        if atom_order[0] == b and atom_order[-1] == a:
            return list(reversed(atom_order))

    path = list(Chem.rdmolops.GetShortestPath(mol, int(a), int(b)))
    if len(path) == 4:
        return [int(each) for each in path]
    raise ValueError(f"Cannot find a 4pi path between atoms {a} and {b}")

def set_diene_bond_pattern(rw_mol, diene_atoms):
    """Convert double-single-double into the product single-double-single pattern."""
    target_bonds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.SINGLE]
    for atom_idx in diene_atoms:
        rw_mol.GetAtomWithIdx(int(atom_idx)).SetIsAromatic(False)
    for atom_i, atom_j, bond_type in zip(
        diene_atoms[:-1], diene_atoms[1:], target_bonds
    ):
        bond = rw_mol.GetBondBetweenAtoms(int(atom_i), int(atom_j))
        if bond is None:
            raise ValueError(f"Cannot find diene bond {atom_i}-{atom_j}")
        bond.SetIsAromatic(False)
        bond.SetBondType(bond_type)

def make_methanethiolate_product_mol(
    mol, aligned_position, fragment_position, a, b, attack_label
):
    attack_atom = a if attack_label == "a" else b
    partner_atom = b if attack_label == "a" else a
    diene_atoms = find_diene_atom_order(mol, a, b)

    rw_mol = Chem.RWMol(mol)
    set_diene_bond_pattern(rw_mol, diene_atoms)

    sulfur_idx = rw_mol.AddAtom(Chem.Atom("S"))
    methyl_c_idx = rw_mol.AddAtom(Chem.Atom("C"))
    hydrogen_indices = [rw_mol.AddAtom(Chem.Atom("H")) for _ in range(3)]
    rw_mol.AddBond(int(sulfur_idx), int(methyl_c_idx), Chem.BondType.SINGLE)
    for hydrogen_idx in hydrogen_indices:
        rw_mol.AddBond(int(methyl_c_idx), int(hydrogen_idx), Chem.BondType.SINGLE)
    rw_mol.AddBond(int(sulfur_idx), int(attack_atom), Chem.BondType.SINGLE)

    # MeS- adds as a closed-shell anion; the negative charge resides at the
    # other end of the conjugated 4pi system in this product resonance form.
    partner = rw_mol.GetAtomWithIdx(int(partner_atom))
    partner.SetFormalCharge(partner.GetFormalCharge() - 1)

    combined_mol = rw_mol.GetMol()
    combined_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(combined_mol, catchErrors=True)
    combined_position = np.vstack([aligned_position, fragment_position])
    set_conformer_positions(combined_mol, combined_position)
    return (
        combined_mol,
        int(attack_atom),
        int(partner_atom),
        int(sulfur_idx),
        int(methyl_c_idx),
        diene_atoms,
    )

def uff_minimize_positions(mol, distance_constraint=None, max_iters=1000):
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if ff is None:
            return conformer_positions(mol), "uff_unavailable"
        if distance_constraint is not None:
            atom_i, atom_j, distance, window, force_constant = distance_constraint
            ff.UFFAddDistanceConstraint(
                int(atom_i),
                int(atom_j),
                False,
                float(distance),
                float(distance) + float(window),
                float(force_constant),
            )
        ff.Initialize()
        status = ff.Minimize(maxIts=max_iters)
        return conformer_positions(mol), f"uff_status_{status}"
    except Exception as exc:
        return conformer_positions(mol), f"uff_failed:{type(exc).__name__}"

def build_methanethiolate_product_structure(
    symbols,
    mol,
    aligned_position,
    a,
    b,
    attack_label,
    face,
    run_uff=True,
):
    fragment_position = build_methanethiolate_fragment(
        aligned_position, a, b, attack_label, face
    )
    result = make_methanethiolate_product_mol(
        mol, aligned_position, fragment_position, a, b, attack_label
    )
    combined_mol, attack_atom, partner_atom, sulfur_idx, methyl_c_idx, diene_atoms = result

    new_symbols = list(symbols) + ["S", "C", "H", "H", "H"]
    new_position = np.vstack([aligned_position, fragment_position])
    uff_status = "skipped"
    if run_uff:
        constraint = (
            attack_atom,
            sulfur_idx,
            PRODUCT_CS_DISTANCE,
            0.01,
            200.0,
        )
        new_position, uff_status = uff_minimize_positions(
            combined_mol, distance_constraint=constraint
        )
        set_conformer_positions(combined_mol, new_position)

    face_flag = 1 if face == "up" else -1
    rel_a, rel_b = (0, 1) if attack_label == "a" else (1, 0)
    title = " ".join(
        str(number) for number in [a, b, rel_a, rel_b, sulfur_idx, face_flag]
    )
    return {
        "symbols": new_symbols,
        "position": new_position,
        "title": title,
        "uff_status": uff_status,
        "diene_atoms": diene_atoms,
        "mol": combined_mol,
        "attack_atom": attack_atom,
        "partner_atom": partner_atom,
        "sulfur_idx": sulfur_idx,
        "methyl_c_idx": methyl_c_idx,
    }

MES_CASES = (("a", "up"), ("a", "down"), ("b", "up"), ("b", "down"))

# Original 7_MAA_calc_MeS.ipynb, zero-based cell 5.
def row_key(row):
    if "index" in row and not pd.isna(row["index"]):
        return to_int(row["index"])
    return to_int(row.name)

def write_methanethiolate_reference_gjf(
    output_dir=OUTPUT_DIR, method=PRODUCT_METHOD, overwrite=False
):
    """Write a closed-shell CH3S- reference input."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gjf_path = output_dir / "MeS_reference.gjf"
    if gjf_path.exists() and not overwrite:
        return gjf_path

    sulfur = np.array([0.0, 0.0, 0.0])
    methyl_c = np.array([0.0, 0.0, SC_DISTANCE])
    c_to_s = unit_vector(sulfur - methyl_c)
    u, v = perpendicular_basis(c_to_s)
    axial = math.cos(math.radians(TETRAHEDRAL_ANGLE_DEG))
    radial = math.sin(math.radians(TETRAHEDRAL_ANGLE_DEG))
    hydrogens = []
    for azimuth_deg in (0.0, 120.0, 240.0):
        phi = math.radians(azimuth_deg)
        direction = axial * c_to_s + radial * (
            math.cos(phi) * u + math.sin(phi) * v
        )
        hydrogens.append(methyl_c + CH_DISTANCE * direction)

    symbols = ["S", "C", "H", "H", "H"]
    position = np.vstack([sulfur, methyl_c, *hydrogens])
    format_change.block_to_gjf(
        symbols, position, str(gjf_path), -1, "MeS_reference", method=method
    )
    return gjf_path

def build_row_methanethiolate_product_gjfs(
    row, stable_table, output_dir=OUTPUT_DIR, method=PRODUCT_METHOD, overwrite=False
):
    diene_index = to_int(row["Diene_Index"])
    stable_conf_id = get_stable_conf_id(diene_index, stable_table)
    a, b = parse_reactive_sites(row["Title"])
    symbols, position, charge, mol, log_path, mol_path = read_diene_structure(
        diene_index, stable_conf_id
    )
    aligned_position, alignment_atom = align_reactive_site(position, mol, a, b)

    output_dir = Path(output_dir)
    product_dir = output_dir / "product"
    mol_output_dir = output_dir / "mol"
    product_dir.mkdir(parents=True, exist_ok=True)
    mol_output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    key = row_key(row)
    for attack_label, face in MES_CASES:
        built = build_methanethiolate_product_structure(
            symbols, mol, aligned_position, a, b, attack_label, face
        )
        attack_atom = built["attack_atom"]
        case_name = f"{attack_label}{attack_atom}_{face}"
        stem = f"MAA_MeS_D{diene_index:05d}_R{key:06d}_{case_name}"
        gjf_path = product_dir / f"{stem}.gjf"
        product_mol_path = mol_output_dir / f"{stem}.mol"

        written = False
        if overwrite or not gjf_path.exists():
            format_change.block_to_gjf(
                built["symbols"],
                built["position"],
                str(gjf_path),
                charge - 1,
                built["title"],
                method=method,
            )
            Chem.MolToMolFile(built["mol"], str(product_mol_path))
            written = True

        records.append({
            "row_key": key,
            "diene_index": diene_index,
            "stable_conf_id": stable_conf_id,
            "diene_smiles": row.get("Diene", ""),
            "a": a,
            "b": b,
            "alignment_atom": alignment_atom,
            "diene_atoms": " ".join(str(each) for each in built["diene_atoms"]),
            "attack_label": attack_label,
            "attack_atom": attack_atom,
            "partner_atom": built["partner_atom"],
            "sulfur_idx": built["sulfur_idx"],
            "methyl_c_idx": built["methyl_c_idx"],
            "face": face,
            "title": built["title"],
            "charge": charge - 1,
            "uff_status": built["uff_status"],
            "gjf_path": str(gjf_path),
            "product_mol_path": str(product_mol_path),
            "source_log_path": str(log_path),
            "source_mol_path": str(mol_path),
            "written": written,
        })
    return records

def generate_methanethiolate_product_inputs(
    input_csv=INPUT_CSV,
    all_smiles_csv=ALL_SMILES_CSV,
    output_dir=OUTPUT_DIR,
    method=PRODUCT_METHOD,
    start=0,
    limit=None,
    overwrite=False,
    write_reference=True,
):
    input_df = pd.read_csv(input_csv)
    stable_table = load_stable_conf_table(all_smiles_csv)
    stop = None if limit is None else start + limit
    work_df = input_df.iloc[start:stop]

    records, failures = [], []
    if write_reference:
        write_methanethiolate_reference_gjf(
            output_dir=output_dir, overwrite=overwrite
        )

    for _, row in tqdm(work_df.iterrows(), total=len(work_df)):
        try:
            records.extend(
                build_row_methanethiolate_product_gjfs(
                    row,
                    stable_table,
                    output_dir=output_dir,
                    method=method,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            failures.append({
                "row_key": row_key(row) if "index" in row else row.name,
                "diene_index": row.get("Diene_Index", None),
                "title": row.get("Title", None),
                "error": repr(exc),
            })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(records)
    failed = pd.DataFrame(failures)
    if len(manifest):
        manifest.to_csv(output_dir / "manifest.csv", index=False)
    if len(failed):
        failed.to_csv(output_dir / "failed.csv", index=False)
    print(f"Wrote/checked {len(manifest)} product input records in {output_dir}")
    if len(failed):
        print(f"Failed rows: {len(failed)}. See {output_dir / 'failed.csv'}")
    return manifest, failed

def title_to_string(title):
    if isinstance(title, (list, tuple, np.ndarray)):
        return " ".join(str(int(each)) for each in title)
    return " ".join(str(title).split())

def parse_title_numbers(title):
    parts = title_to_string(title).split()
    if len(parts) < 5:
        raise ValueError(f"Title should contain at least five numbers: {title!r}")
    return [to_int(each) for each in parts]

def reaction_atoms_from_title(title):
    """Return attack atom, partner atom, and sulfur index."""
    numbers = parse_title_numbers(title)
    a, b, rel_a, rel_b, sulfur_idx = numbers[:5]
    if rel_a == 0:
        attack_atom, partner_atom = a, b
    elif rel_b == 0:
        attack_atom, partner_atom = b, a
    else:
        raise ValueError(
            f"Cannot determine MeS attack atom from title: {title_to_string(title)}"
        )
    return attack_atom, partner_atom, sulfur_idx, numbers

def read_product_log_geometry(product_log_path):
    parsed = logfile_process.Logfile(str(product_log_path))
    symbols = normalize_symbols(parsed.symbol_list)
    position = final_log_position(parsed)
    charge = int(parsed.charge)
    title = title_to_string(parsed.title)
    return symbols, position, charge, title

def product_log_to_updated_mol(product_log_path, product_mol_path):
    symbols, position, charge, title = read_product_log_geometry(product_log_path)
    product_mol = Chem.MolFromMolFile(
        str(product_mol_path), removeHs=False, sanitize=False
    )
    if product_mol is None:
        raise ValueError(f"RDKit failed to read product mol: {product_mol_path}")
    if product_mol.GetNumAtoms() != len(position):
        raise ValueError(
            f"Mol/log atom count mismatch: mol={product_mol.GetNumAtoms()}, "
            f"log={len(position)}"
        )
    set_conformer_positions(product_mol, position)
    return product_mol, symbols, position, charge, title

def fragment_atoms_from_anchor(mol, anchor_atom, blocked_atom):
    """Return the anchor-side fragment after cutting blocked_atom-anchor_atom."""
    visited = set()
    stack = [int(anchor_atom)]
    blocked_atom = int(blocked_atom)
    while stack:
        atom_idx = stack.pop()
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if atom_idx == int(anchor_atom) and neighbor_idx == blocked_atom:
                continue
            if neighbor_idx not in visited:
                stack.append(neighbor_idx)
    return sorted(visited)

def distance_between(position, atom_i, atom_j):
    position = np.asarray(position, dtype=float)
    return float(np.linalg.norm(position[int(atom_i)] - position[int(atom_j)]))

def stretch_bond_to_distance(
    position, fixed_atom, mobile_atom, target_distance, move_atoms
):
    """Translate a whole fragment to set one bond distance exactly."""
    new_position = np.asarray(position, dtype=float).copy()
    fixed_atom = int(fixed_atom)
    mobile_atom = int(mobile_atom)
    vector = new_position[mobile_atom] - new_position[fixed_atom]
    current_distance = float(np.linalg.norm(vector))
    if current_distance <= 1.0e-12:
        raise ValueError(f"Cannot stretch zero-length bond {fixed_atom}-{mobile_atom}")

    target_mobile = (
        new_position[fixed_atom]
        + vector / current_distance * float(target_distance)
    )
    shift = target_mobile - new_position[mobile_atom]
    for atom_idx in move_atoms:
        new_position[int(atom_idx)] += shift
    final_distance = distance_between(new_position, fixed_atom, mobile_atom)
    return new_position, current_distance, final_distance

def scan_scale_factors(
    start=SCAN_FACTOR_START, stop=SCAN_FACTOR_STOP, step=SCAN_FACTOR_STEP
):
    """Return decimal-exact factors including both endpoints."""
    start_d, stop_d, step_d = map(
        lambda value: Decimal(str(value)), (start, stop, step)
    )
    if step_d <= 0 or stop_d < start_d:
        raise ValueError("Require step > 0 and stop >= start")
    factors = []
    value = start_d
    while value <= stop_d:
        factors.append(float(value))
        value += step_d
    return factors

def safe_distance_tag(distance):
    return f"{float(distance):.4f}".replace(".", "p")

def product_log_to_cs_scan_gjfs(
    product_log_path,
    product_mol_dir=OUTPUT_DIR / "mol",
    scan_dir=OUTPUT_DIR / "ts_scan",
    factor_start=SCAN_FACTOR_START,
    factor_stop=SCAN_FACTOR_STOP,
    factor_step=SCAN_FACTOR_STEP,
    mode="ts",
    overwrite=False,
):
    """Convert one optimized product into a 1.00--1.50 x d0 C-S series.

    mode="ts" writes an independent TS optimization at every point.
    mode="constrained" writes a fixed-distance relaxed optimization at every point.
    """
    if mode not in {"ts", "constrained"}:
        raise ValueError("mode must be 'ts' or 'constrained'")

    product_log_path = Path(product_log_path)
    product_mol_path = Path(product_mol_dir) / f"{product_log_path.stem}.mol"
    if not product_mol_path.exists():
        raise FileNotFoundError(
            f"Cannot find product mol for {product_log_path.name}: {product_mol_path}"
        )

    product_mol, symbols, position, charge, title = product_log_to_updated_mol(
        product_log_path, product_mol_path
    )
    attack_atom, partner_atom, sulfur_idx, title_numbers = reaction_atoms_from_title(
        title
    )
    mes_fragment = fragment_atoms_from_anchor(
        product_mol, sulfur_idx, attack_atom
    )
    if sulfur_idx not in mes_fragment or attack_atom in mes_fragment:
        raise ValueError("Failed to isolate the S-CH3 mobile fragment")

    d0 = distance_between(position, attack_atom, sulfur_idx)
    factors = scan_scale_factors(factor_start, factor_stop, factor_step)
    scan_dir = Path(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for point_id, factor in enumerate(factors):
        target_distance = d0 * factor
        scan_position, _, final_distance = stretch_bond_to_distance(
            position,
            attack_atom,
            sulfur_idx,
            target_distance,
            mes_fragment,
        )
        factor_tag = f"{factor:.2f}".replace(".", "p")
        distance_tag = safe_distance_tag(target_distance)
        gjf_path = scan_dir / (
            f"{product_log_path.stem}_p{point_id:02d}"
            f"_f{factor_tag}_d{distance_tag}.gjf"
        )

        method = TS_METHOD if mode == "ts" else RELAXED_SCAN_METHOD
        final_line = None
        if mode == "constrained":
            # Gaussian ModRedundant atom indices are 1-based.
            final_line = f"B {attack_atom + 1} {sulfur_idx + 1} F"

        written = False
        if overwrite or not gjf_path.exists():
            format_change.block_to_gjf(
                symbols,
                scan_position,
                str(gjf_path),
                charge,
                title,
                method=method,
                final_line=final_line,
            )
            written = True

        records.append({
            "product_log_path": str(product_log_path),
            "product_mol_path": str(product_mol_path),
            "scan_gjf_path": str(gjf_path),
            "mode": mode,
            "point_id": point_id,
            "factor": factor,
            "product_cs_distance": d0,
            "target_cs_distance": target_distance,
            "final_cs_distance": final_distance,
            "attack_atom": attack_atom,
            "partner_atom": partner_atom,
            "sulfur_idx": sulfur_idx,
            "mes_fragment": " ".join(str(each) for each in mes_fragment),
            "charge": charge,
            "title": title,
            "written": written,
        })

    return records

def generate_methanethiolate_ts_scan_inputs(
    product_log_dir=OUTPUT_DIR / "product",
    product_mol_dir=OUTPUT_DIR / "mol",
    scan_dir=OUTPUT_DIR / "ts_scan",
    factor_start=SCAN_FACTOR_START,
    factor_stop=SCAN_FACTOR_STOP,
    factor_step=SCAN_FACTOR_STEP,
    mode="ts",
    start=0,
    limit=None,
    overwrite=False,
):
    product_logs = sorted(Path(product_log_dir).glob("*.log"))
    stop = None if limit is None else start + limit
    product_logs = product_logs[start:stop]

    records, failures = [], []
    for product_log_path in tqdm(product_logs, total=len(product_logs)):
        try:
            records.extend(
                product_log_to_cs_scan_gjfs(
                    product_log_path,
                    product_mol_dir=product_mol_dir,
                    scan_dir=scan_dir,
                    factor_start=factor_start,
                    factor_stop=factor_stop,
                    factor_step=factor_step,
                    mode=mode,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            failures.append({
                "product_log_path": str(product_log_path),
                "error": repr(exc),
            })

    scan_dir = Path(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(records)
    failed = pd.DataFrame(failures)
    if len(manifest):
        manifest.to_csv(scan_dir / "manifest.csv", index=False)
    if len(failed):
        failed.to_csv(scan_dir / "failed.csv", index=False)
    print(
        f"Wrote/checked {len(manifest)} scan inputs from "
        f"{len(product_logs)} product logs in {scan_dir}"
    )
    if len(failed):
        print(f"Failed products: {len(failed)}. See {scan_dir / 'failed.csv'}")
    return manifest, failed

# Original 7_MAA_calc_MeS.ipynb, zero-based cell 12.
def parse_scan_name_distances(stem):
    """Infer scan factor and target C-S distance from a flattened TS stem."""
    import re

    match = re.search(r"_p(?P<point>\d+)_f(?P<factor>\d+p\d+)_d(?P<distance>\d+p\d+)$", stem)
    if match is None:
        return None
    factor = float(match.group("factor").replace("p", "."))
    target_distance = float(match.group("distance").replace("p", "."))
    product_distance = target_distance / factor
    return {
        "point_id": int(match.group("point")),
        "factor": factor,
        "target_cs_distance": target_distance,
        "product_cs_distance": product_distance,
    }

def load_ts_scan_distance_lookup(manifest_csv):
    """Map TS stems to product C-S distances, preferring ts_scan/manifest.csv."""
    manifest_csv = Path(manifest_csv)
    lookup = {}
    if not manifest_csv.exists():
        return lookup

    manifest = pd.read_csv(manifest_csv)
    if "scan_gjf_path" not in manifest.columns or "product_cs_distance" not in manifest.columns:
        return lookup

    for _, row in manifest.iterrows():
        stem = Path(row["scan_gjf_path"]).stem
        lookup[stem] = {
            "product_cs_distance": float(row["product_cs_distance"]),
            "target_cs_distance": float(row.get("target_cs_distance", np.nan)),
            "factor": float(row.get("factor", np.nan)),
            "point_id": int(row.get("point_id", -1)),
        }
    return lookup

def irc_endpoint_distance(irc_log_path, attack_atom, sulfur_idx):
    """Return the C-S distance at the final IRC geometry."""
    parsed = logfile_process.Logfile(str(irc_log_path))
    if parsed.running_positions is None or len(parsed.running_positions) == 0:
        raise ValueError(f"No IRC geometry found in {irc_log_path}")
    position = np.asarray(parsed.running_positions[-1], dtype=float)
    return distance_between(position, attack_atom, sulfur_idx)

def paired_irc_logs(ts_stem, irc_dir):
    irc_dir = Path(irc_dir)
    return {
        "forward": irc_dir / f"{ts_stem}forward.log",
        "reverse": irc_dir / f"{ts_stem}reverse.log",
    }

def move_ts_and_irc_files(ts_stem, ts_dir, irc_dir, success_dir, overwrite=False, dry_run=True):
    """Move TS files and paired IRC files into one flat ts_success folder."""
    import shutil

    ts_dir = Path(ts_dir)
    irc_dir = Path(irc_dir)
    success_dir = Path(success_dir)
    success_dir.mkdir(parents=True, exist_ok=True)

    source_files = []
    source_files.extend(sorted(ts_dir.glob(f"{ts_stem}.*")))
    source_files.extend(sorted(irc_dir.glob(f"{ts_stem}forward.*")))
    source_files.extend(sorted(irc_dir.glob(f"{ts_stem}reverse.*")))

    moved = []
    for source in source_files:
        destination = success_dir / source.name
        if destination.exists() and not overwrite:
            action = "exists_skipped"
        elif dry_run:
            action = "dry_run"
        else:
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
            action = "moved"
        moved.append({
            "source_path": str(source),
            "destination_path": str(destination),
            "action": action,
        })
    return moved

def select_successful_methanethiolate_irc(
    output_dir=OUTPUT_DIR,
    ts_name="ts_scan",
    irc_name="irc_scan",
    success_name="ts_success",
    lower_factor=0.75,
    upper_factor=1.25,
    overwrite=False,
    dry_run=True,
):
    """Select TS structures whose IRC connects product and separated endpoints.

    A TS is marked successful when exactly one IRC endpoint has the 4pi-C--S
    distance within [0.75, 1.25] * product_C-S_distance and the other endpoint
    is longer than 1.25 * product_C-S_distance.
    """
    output_dir = Path(output_dir)
    ts_dir = output_dir / ts_name
    irc_dir = output_dir / irc_name
    success_dir = output_dir / success_name

    ts_logs = sorted(ts_dir.glob("*.log"))
    distance_lookup = load_ts_scan_distance_lookup(ts_dir / "manifest.csv")
    records, failures, moved_records = [], [], []

    for ts_log_path in tqdm(ts_logs, total=len(ts_logs)):
        ts_stem = ts_log_path.stem
        try:
            ts_log = logfile_process.Logfile(str(ts_log_path))
            attack_atom, partner_atom, sulfur_idx, title_numbers = reaction_atoms_from_title(ts_log.title)

            distance_record = distance_lookup.get(ts_stem)
            if distance_record is None:
                distance_record = parse_scan_name_distances(ts_stem)
            if distance_record is None:
                raise ValueError(f"Cannot determine product C-S distance for {ts_stem}")

            product_distance = float(distance_record["product_cs_distance"])
            lower_cutoff = lower_factor * product_distance
            upper_cutoff = upper_factor * product_distance

            irc_logs = paired_irc_logs(ts_stem, irc_dir)
            missing = [name for name, path in irc_logs.items() if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing IRC logs for {ts_stem}: {missing}")

            endpoint_distances = {
                name: irc_endpoint_distance(path, attack_atom, sulfur_idx)
                for name, path in irc_logs.items()
            }
            connected = {
                name: lower_cutoff <= distance <= upper_cutoff
                for name, distance in endpoint_distances.items()
            }
            separated = {
                name: distance > upper_cutoff
                for name, distance in endpoint_distances.items()
            }
            success = (
                sum(connected.values()) == 1
                and sum(separated.values()) == 1
                and all(connected[name] != separated[name] for name in endpoint_distances)
            )

            moved_files = []
            if success:
                moved_files = move_ts_and_irc_files(
                    ts_stem,
                    ts_dir=ts_dir,
                    irc_dir=irc_dir,
                    success_dir=success_dir,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
                for moved in moved_files:
                    moved_record = {"ts_stem": ts_stem, **moved}
                    moved_records.append(moved_record)

            records.append({
                "ts_stem": ts_stem,
                "ts_log_path": str(ts_log_path),
                "attack_atom": attack_atom,
                "partner_atom": partner_atom,
                "sulfur_idx": sulfur_idx,
                "product_cs_distance": product_distance,
                "lower_cutoff": lower_cutoff,
                "upper_cutoff": upper_cutoff,
                "forward_distance": endpoint_distances["forward"],
                "reverse_distance": endpoint_distances["reverse"],
                "forward_connected": connected["forward"],
                "reverse_connected": connected["reverse"],
                "forward_separated": separated["forward"],
                "reverse_separated": separated["reverse"],
                "success": success,
                "moved_file_count": len(moved_files),
            })
        except Exception as exc:
            failures.append({
                "ts_stem": ts_stem,
                "ts_log_path": str(ts_log_path),
                "error": repr(exc),
            })

    success_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    failed = pd.DataFrame(failures)
    moved = pd.DataFrame(moved_records)
    result.to_csv(success_dir / "ts_success_manifest.csv", index=False)
    if len(failed):
        failed.to_csv(success_dir / "ts_success_failed.csv", index=False)
    if len(moved):
        moved.to_csv(success_dir / "ts_success_moved_files.csv", index=False)

    success_count = int(result["success"].sum()) if len(result) else 0
    print(
        f"Checked {len(result)} TS logs; selected {success_count} successful TS. "
        f"dry_run={dry_run}. Results: {success_dir}"
    )
    if len(failed):
        print(f"Failed checks: {len(failed)}. See {success_dir / 'ts_success_failed.csv'}")
    return result, failed, moved

# Original 7_MAA_calc_MeS.ipynb, zero-based cell 14.
def success_ts_stems(success_dir):
    """Return successful TS stems, excluding paired IRC forward/reverse logs."""
    success_dir = Path(success_dir)
    manifest_csv = success_dir / "ts_success_manifest.csv"
    if manifest_csv.exists():
        manifest = pd.read_csv(manifest_csv)
        if "success" in manifest.columns and "ts_stem" in manifest.columns:
            success_mask = manifest["success"].astype(bool)
            return sorted(set(str(each) for each in manifest.loc[success_mask, "ts_stem"]))

    stems = []
    for log_path in sorted(success_dir.glob("*.log")):
        stem = log_path.stem
        if stem.endswith("forward") or stem.endswith("reverse"):
            continue
        stems.append(stem)
    return stems

def write_success_ts_spe_inputs(
    output_dir=OUTPUT_DIR,
    success_name="ts_success",
    eng_name="ts_success_eng",
    method=ENG_METHOD,
    overwrite=False,
):
    """Write solvent single-point inputs for successful TS logs only."""
    output_dir = Path(output_dir)
    success_dir = output_dir / success_name
    eng_dir = output_dir / eng_name
    eng_dir.mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    for stem in tqdm(success_ts_stems(success_dir)):
        ts_log_path = success_dir / f"{stem}.log"
        spe_gjf_path = eng_dir / f"{stem}.gjf"
        try:
            if not ts_log_path.exists():
                raise FileNotFoundError(f"Missing successful TS log: {ts_log_path}")
            ts_log = logfile_process.Logfile(str(ts_log_path))
            if ts_log.running_positions is None or len(ts_log.running_positions) == 0:
                raise ValueError(f"No final TS geometry found in {ts_log_path}")

            title = title_to_string(ts_log.title)
            written = False
            if overwrite or not spe_gjf_path.exists():
                format_change.block_to_gjf(
                    ts_log.symbol_list,
                    ts_log.running_positions[-1],
                    str(spe_gjf_path),
                    ts_log.charge,
                    title,
                    method=method,
                )
                written = True

            records.append({
                "ts_stem": stem,
                "ts_log_path": str(ts_log_path),
                "spe_gjf_path": str(spe_gjf_path),
                "charge": int(ts_log.charge),
                "title": title,
                "method": method,
                "written": written,
            })
        except Exception as exc:
            failures.append({
                "ts_stem": stem,
                "ts_log_path": str(ts_log_path),
                "spe_gjf_path": str(spe_gjf_path),
                "error": repr(exc),
            })

    manifest = pd.DataFrame(records)
    failed = pd.DataFrame(failures)
    if len(manifest):
        manifest.to_csv(eng_dir / "manifest.csv", index=False)
    if len(failed):
        failed.to_csv(eng_dir / "failed.csv", index=False)
    print(f"Wrote/checked {len(manifest)} successful TS SPE inputs in {eng_dir}")
    if len(failed):
        print(f"Failed successful TS SPE inputs: {len(failed)}. See {eng_dir / 'failed.csv'}")
    return manifest, failed

SPE_SOLVENT_METHOD = "b3lyp/6-311+g(d,p) em=gd3bj scrf=(smd, solvent=water)"

# Original 7_MAA_calc_MeS.ipynb, zero-based cell 16.
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

MES_DISTORT_METHOD = "b3lyp/6-311+g(d,p) em=gd3bj"

def read_opt_spe_total_g(opt_log, eng_log=None):
    """Return SPE electronic energy plus opt/freq thermal Gibbs correction."""
    opt_log = Path(opt_log)
    opt = logfile_process.Logfile(str(opt_log))
    thermal_g_corr = float(opt.all_engs[-1])
    if eng_log is not None and Path(eng_log).exists():
        eng = logfile_process.Logfile(str(eng_log))
        return float(eng.all_engs[-1]) + thermal_g_corr
    if hasattr(opt, "all_engs") and len(opt.all_engs) >= 2:
        return float(opt.all_engs[0]) + thermal_g_corr
    raise ValueError(f"Cannot read total G from {opt_log}")

def read_methanethiolate_reference_g(output_dir=OUTPUT_DIR, mes_g_hartree=None):
    """Read MeS- reference G; fall back to zero shift for correlation-only work."""
    if mes_g_hartree is not None:
        return float(mes_g_hartree), "provided"

    output_dir = Path(output_dir)
    opt_log = output_dir / "MeS_reference.log"
    eng_log = output_dir / "MeS_reference_eng.log"
    if opt_log.exists():
        return read_opt_spe_total_g(opt_log, eng_log if eng_log.exists() else None), "read_reference"

    print(
        "WARNING: MeS_reference.log was not found. Using MeS_G_Hartree = 0.0; "
        "deltaG/deltaGa are uniformly shifted and should be used only for correlations until the reference is added."
    )
    return 0.0, "missing_reference_zero_shift"

def infer_product_log_path(gjf_path):
    return Path(gjf_path).with_suffix(".log")

def infer_product_eng_log_path(product_log_path, output_dir=OUTPUT_DIR):
    return Path(output_dir) / "product_eng" / Path(product_log_path).name

def product_stem_from_ts_stem(ts_stem):
    import re

    return re.sub(r"_p\d+_f\d+p\d+_d\d+p\d+$", "", str(ts_stem))

def read_input_descriptor_table(input_csv=INPUT_CSV):
    df = pd.read_csv(input_csv).copy()
    df["Diene_Index"] = df["Diene_Index"].map(to_int)
    return df

def collect_mes_product_thermodynamics(
    manifest_csv=OUTPUT_DIR / "manifest.csv",
    input_csv=INPUT_CSV,
    all_smiles_csv=ALL_SMILES_CSV,
    output_dir=OUTPUT_DIR,
    mes_g_hartree=None,
    analysis_dir=ANALYSIS_DIR,
):
    mes_g_hartree, mes_ref_status = read_methanethiolate_reference_g(output_dir, mes_g_hartree)
    manifest = pd.read_csv(manifest_csv).copy()
    target_df = read_input_descriptor_table(input_csv)
    target_by_diene = target_df.drop_duplicates("Diene_Index").set_index("Diene_Index")
    stable_table = pd.read_csv(all_smiles_csv, index_col="Index")

    rows = []
    for _, item in tqdm(manifest.iterrows(), total=len(manifest)):
        diene_index = to_int(item["diene_index"])
        product_log = infer_product_log_path(item["gjf_path"])
        product_eng_log = infer_product_eng_log_path(product_log, output_dir=output_dir)

        result = item.to_dict()
        result.update({
            "product_stem": Path(item["gjf_path"]).stem,
            "product_log_path": str(product_log),
            "product_eng_log_path": str(product_eng_log),
            "MeS_G_Hartree": float(mes_g_hartree),
            "MeS_reference_status": mes_ref_status,
            "product_G_solv_Hartree": np.nan,
            "diene_G_solv_Hartree": np.nan,
            "MeS_deltaG": np.nan,
            "thermo_status": "missing_log",
        })

        if product_log.exists() and product_eng_log.exists():
            product_g_solv = read_opt_spe_total_g(product_log, product_eng_log)
            diene_g_solv = float(stable_table.loc[diene_index, "G(Solvent)/Hatree"])
            result.update({
                "product_G_solv_Hartree": product_g_solv,
                "diene_G_solv_Hartree": diene_g_solv,
                "MeS_deltaG": HARTREE_TO_KCAL * (product_g_solv - diene_g_solv - float(mes_g_hartree)),
                "thermo_status": "ok",
            })

        if diene_index in target_by_diene.index:
            for col in ["deltaG", "deltaGa", "deltaGa(Solvent)", "Diene_Distort", "Ene_Distort", "Interaction", "Diene", "Ene"]:
                if col in target_by_diene.columns:
                    result[f"DA_{col}" if col in {"deltaG", "deltaGa"} else col] = target_by_diene.loc[diene_index, col]
        rows.append(result)

    all_df = pd.DataFrame(rows)
    completed = all_df.dropna(subset=["MeS_deltaG"]).copy()
    best_df = (
        completed.sort_values("MeS_deltaG")
        .groupby("diene_index", as_index=False)
        .first()
        if len(completed)
        else pd.DataFrame()
    )

    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(analysis_dir / "MeS_product_thermodynamics_all.csv", index=False)
    if len(best_df):
        best_df.to_csv(analysis_dir / "MeS_product_thermodynamics_best.csv", index=False)

    print(f"Collected {len(completed)} completed MeS product thermodynamics")
    print(f"Selected {len(best_df)} lowest-deltaG products by diene")
    return all_df, best_df

def collect_mes_success_ts_kinetics(
    success_manifest_csv=OUTPUT_DIR / "ts_success" / "ts_success_manifest.csv",
    product_manifest_csv=OUTPUT_DIR / "manifest.csv",
    input_csv=INPUT_CSV,
    all_smiles_csv=ALL_SMILES_CSV,
    output_dir=OUTPUT_DIR,
    mes_g_hartree=None,
    analysis_dir=ANALYSIS_DIR,
):
    mes_g_hartree, mes_ref_status = read_methanethiolate_reference_g(output_dir, mes_g_hartree)
    success_manifest = pd.read_csv(success_manifest_csv).copy()
    success_manifest = success_manifest[success_manifest["success"].astype(bool)].copy()
    product_manifest = pd.read_csv(product_manifest_csv).copy()
    product_manifest["product_stem"] = product_manifest["gjf_path"].map(lambda path: Path(path).stem)
    product_by_stem = product_manifest.set_index("product_stem")
    target_df = read_input_descriptor_table(input_csv)
    target_by_diene = target_df.drop_duplicates("Diene_Index").set_index("Diene_Index")
    stable_table = pd.read_csv(all_smiles_csv, index_col="Index")

    rows = []
    for _, item in tqdm(success_manifest.iterrows(), total=len(success_manifest)):
        ts_stem = str(item["ts_stem"])
        product_stem = product_stem_from_ts_stem(ts_stem)
        product_item = product_by_stem.loc[product_stem] if product_stem in product_by_stem.index else pd.Series(dtype=object)
        diene_index = to_int(product_item.get("diene_index", np.nan)) if len(product_item) else np.nan
        ts_log = Path(output_dir) / "ts_success" / f"{ts_stem}.log"
        if not ts_log.exists() and "ts_log_path" in item:
            ts_log = Path(item["ts_log_path"])
        ts_eng_log = Path(output_dir) / "ts_success_eng" / f"{ts_stem}.log"

        result = item.to_dict()
        result.update({
            "product_stem": product_stem,
            "diene_index": diene_index,
            "ts_success_log_path": str(ts_log),
            "ts_success_eng_log_path": str(ts_eng_log),
            "MeS_G_Hartree": float(mes_g_hartree),
            "MeS_reference_status": mes_ref_status,
            "ts_G_solv_Hartree": np.nan,
            "diene_G_solv_Hartree": np.nan,
            "MeS_deltaGa": np.nan,
            "kinetic_status": "missing_log",
        })

        if not pd.isna(diene_index) and ts_log.exists() and ts_eng_log.exists():
            ts_g_solv = read_opt_spe_total_g(ts_log, ts_eng_log)
            diene_g_solv = float(stable_table.loc[to_int(diene_index), "G(Solvent)/Hatree"])
            result.update({
                "ts_G_solv_Hartree": ts_g_solv,
                "diene_G_solv_Hartree": diene_g_solv,
                "MeS_deltaGa": HARTREE_TO_KCAL * (ts_g_solv - diene_g_solv - float(mes_g_hartree)),
                "kinetic_status": "ok",
            })

        if not pd.isna(diene_index) and to_int(diene_index) in target_by_diene.index:
            for col in ["deltaG", "deltaGa", "deltaGa(Solvent)", "Diene_Distort", "Ene_Distort", "Interaction", "Diene", "Ene"]:
                if col in target_by_diene.columns:
                    result[f"DA_{col}" if col in {"deltaG", "deltaGa"} else col] = target_by_diene.loc[to_int(diene_index), col]
        rows.append(result)

    all_df = pd.DataFrame(rows)
    completed = all_df.dropna(subset=["MeS_deltaGa"]).copy()
    best_df = (
        completed.sort_values("MeS_deltaGa")
        .groupby("diene_index", as_index=False)
        .first()
        if len(completed)
        else pd.DataFrame()
    )

    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(analysis_dir / "MeS_success_ts_kinetics_all.csv", index=False)
    if len(best_df):
        best_df.to_csv(analysis_dir / "MeS_success_ts_kinetics_best.csv", index=False)

    print(f"Collected {len(completed)} completed successful-TS kinetics")
    print(f"Selected {len(best_df)} lowest-deltaGa successful TS by diene")
    return all_df, best_df

def build_mes_thermo_kinetic_summary(
    product_best_df,
    ts_best_df,
    analysis_dir=ANALYSIS_DIR,
):
    left_cols = [
        "diene_index", "product_stem", "MeS_deltaG", "product_log_path", "product_eng_log_path",
        "Diene_Distort", "Ene_Distort", "Interaction", "deltaGa(Solvent)", "Diene", "Ene",
    ]
    right_cols = [
        "diene_index", "ts_stem", "product_stem", "MeS_deltaGa", "ts_success_log_path",
        "ts_success_eng_log_path", "product_cs_distance", "forward_distance", "reverse_distance",
    ]
    thermo = product_best_df[[col for col in left_cols if col in product_best_df.columns]].copy()
    kinetic = ts_best_df[[col for col in right_cols if col in ts_best_df.columns]].copy()
    summary = thermo.merge(kinetic, on="diene_index", how="outer", suffixes=("_thermo", "_kinetic"))
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis_dir / "MeS_thermo_kinetic_summary.csv", index=False)
    return summary

def product_log_to_mes_distorted_diene_gjf(
    product_log_path,
    distort_dir=OUTPUT_DIR / "product_distort",
    method=MES_DISTORT_METHOD,
    overwrite=False,
):
    product_log_path = Path(product_log_path)
    log = logfile_process.Logfile(str(product_log_path))
    symbols = normalize_symbols(log.symbol_list)
    position = final_log_position(log)
    title = title_to_string(log.title)
    attack_atom, partner_atom, sulfur_idx, title_numbers = reaction_atoms_from_title(title)

    distorted_symbols = symbols[:sulfur_idx]
    distorted_position = position[:sulfur_idx]
    diene_charge = int(log.charge) + 1  # product is diene + MeS-, so remove the anion charge.

    distort_dir = Path(distort_dir)
    distort_dir.mkdir(parents=True, exist_ok=True)
    gjf_path = distort_dir / f"{product_log_path.stem}.gjf"
    if overwrite or not gjf_path.exists():
        format_change.block_to_gjf(
            distorted_symbols,
            distorted_position,
            str(gjf_path),
            diene_charge,
            title,
            method=method,
        )

    return {
        "product_log_path": str(product_log_path),
        "distort_gjf_path": str(gjf_path),
        "kept_atoms": sulfur_idx,
        "removed_atoms": " ".join(str(each) for each in range(sulfur_idx, len(symbols))),
        "diene_charge": diene_charge,
        "title": title,
    }

def generate_mes_selected_diene_distortion_inputs(
    summary_csv=ANALYSIS_DIR / "MeS_thermo_kinetic_summary.csv",
    output_dir=OUTPUT_DIR,
    overwrite=False,
):
    summary_df = pd.read_csv(summary_csv)
    records, failures = [], []
    for _, row in tqdm(summary_df.iterrows(), total=len(summary_df)):
        product_log_path = Path(str(row.get("product_log_path", "")))
        try:
            if not product_log_path.exists():
                raise FileNotFoundError(f"Missing selected product log: {product_log_path}")
            records.append(
                product_log_to_mes_distorted_diene_gjf(
                    product_log_path,
                    distort_dir=Path(output_dir) / "product_distort",
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            failures.append({
                "diene_index": row.get("diene_index", np.nan),
                "product_log_path": str(product_log_path),
                "error": repr(exc),
            })

    manifest = pd.DataFrame(records)
    failed = pd.DataFrame(failures)
    if len(manifest):
        manifest.to_csv(Path(output_dir) / "product_distort_manifest.csv", index=False)
    if len(failed):
        failed.to_csv(Path(output_dir) / "product_distort_failed.csv", index=False)
    print(f"Wrote/checked {len(manifest)} MeS distorted-diene inputs")
    if len(failed):
        print(f"Failed distortion inputs: {len(failed)}")
    return manifest, failed

def collect_mes_diene_distortion_energies(
    summary_csv=ANALYSIS_DIR / "MeS_thermo_kinetic_summary.csv",
    all_smiles_csv=ALL_SMILES_CSV,
    output_dir=OUTPUT_DIR,
):
    final_df = pd.read_csv(summary_csv).copy()
    table = pd.read_csv(all_smiles_csv, index_col="Index")

    for row_id, row in tqdm(final_df.iterrows(), total=len(final_df)):
        diene_index = to_int(row["diene_index"])
        product_log_path = Path(str(row.get("product_log_path", "")))
        distort_log = Path(output_dir) / "product_distort" / f"{product_log_path.stem}.log"
        final_df.loc[row_id, "distort_log_path"] = str(distort_log)
        if not distort_log.exists():
            final_df.loc[row_id, "Distortion_energy"] = np.nan
            continue

        distorted_e = logfile_process.Logfile(str(distort_log)).all_engs[-1]
        relaxed_diene_e = float(table.loc[diene_index, "E/Hatree"])
        final_df.loc[row_id, "Distortion_energy"] = HARTREE_TO_KCAL * (float(distorted_e) - relaxed_diene_e)

    final_df.to_csv(summary_csv, index=False)
    return final_df

def plot_mes_correlation_panel(df, columns=None):
    from matplotlib import pyplot as plt

    contribution_col = "Diene Distortion Contribution"
    display_labels = {
        "Diene_Distort": r"$\Delta E_{\mathrm{dist-4\pi}}$",
        contribution_col: r"$f_{4\pi}$",
        "MeS_deltaG": r"$\Delta G_{\mathrm{rxn,MeS}}$",
        "MeS_deltaGa": r"$\Delta G^{\ddagger}_{\mathrm{MeS}}$",
        "deltaGa(Solvent)": r"$\Delta G^{\ddagger}_{\mathrm{DA}}$",
    }
    axis_labels = {
        "Diene_Distort": r"$\Delta E_{\mathrm{dist-4\pi}}$ (kcal mol$^{-1}$)",
        contribution_col: r"$f_{4\pi}$",
        "MeS_deltaG": r"$\Delta G_{\mathrm{rxn,MeS}}$ (kcal mol$^{-1}$)",
        "MeS_deltaGa": r"$\Delta G^{\ddagger}_{\mathrm{MeS}}$ (kcal mol$^{-1}$)",
    }
    plot_source = df.copy()
    if {"Diene_Distort", "Ene_Distort"}.issubset(plot_source.columns):
        diene_distort = pd.to_numeric(plot_source["Diene_Distort"], errors="coerce")
        ene_distort = pd.to_numeric(plot_source["Ene_Distort"], errors="coerce")
        total_distort = diene_distort + ene_distort
        plot_source[contribution_col] = diene_distort / total_distort.mask(np.isclose(total_distort, 0.0))

    if columns is None:
        columns = ["Diene_Distort", contribution_col, "MeS_deltaG", "MeS_deltaGa", "deltaGa(Solvent)"]
    columns = [col for col in columns if col in plot_source.columns]
    plot_df = plot_source[columns].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if len(plot_df) == 0 or len(columns) < 2:
        print("Not enough numeric data to plot correlations.")
        return pd.DataFrame(), pd.DataFrame()

    pearson = plot_df.corr(method="pearson")
    spearman = plot_df.corr(method="spearman")
    display(pearson.rename(index=display_labels, columns=display_labels))
    display(spearman.rename(index=display_labels, columns=display_labels))

    pair_columns = [
        col for col in ["Diene_Distort", contribution_col, "MeS_deltaG", "MeS_deltaGa"]
        if col in plot_df.columns
    ]
    if len(pair_columns) >= 2:
        fig, axes = plt.subplots(len(pair_columns), len(pair_columns), figsize=(3.2 * len(pair_columns), 3.2 * len(pair_columns)))
        if len(pair_columns) == 1:
            axes = np.array([[axes]])
        for i, y_col in enumerate(pair_columns):
            for j, x_col in enumerate(pair_columns):
                ax = axes[i, j]
                data = plot_df[[x_col, y_col]].dropna()
                if i == j:
                    ax.hist(data[x_col], bins=12)
                else:
                    ax.scatter(data[x_col], data[y_col], s=24)
                    if len(data) >= 2:
                        x = data[x_col].to_numpy(dtype=float)
                        y = data[y_col].to_numpy(dtype=float)
                        slope, intercept = np.polyfit(x, y, 1)
                        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                        ax.plot(xs, slope * xs + intercept)
                if i == len(pair_columns) - 1:
                    ax.set_xlabel(axis_labels.get(x_col, display_labels.get(x_col, x_col)))
                if j == 0:
                    ax.set_ylabel(axis_labels.get(y_col, display_labels.get(y_col, y_col)))
        plt.tight_layout()
        plt.show()

    return pearson, spearman

# Original 7_MAA_calc_MeS.ipynb, zero-based cell 23.
DA_TS_DIRS = [
    SECONDARY_ROOT / "6_DI_selected/ts",
    SECONDARY_ROOT / "6_DI_dropped_18100/ts",
]

DA_ENE_INDEX_FOR_COMPARISON = 18100

def parse_da_ts_filename(log_path):
    import re

    match = re.match(
        r"(?P<diene_index>\d{5})_(?P<ene_index>\d{5})_(?P<structure_id>\d{5})_(?P<conf_id>\d{4})\.log$",
        Path(log_path).name,
    )
    if match is None:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}

def diene_4pi_path(mol, a, b):
    """Return the 4pi path in a -> b order."""
    for atom_order in cycle_process.diene_atom_Idx(Chem.Mol(mol), select_diene=0):
        atom_order = [int(each) for each in atom_order]
        if atom_order[0] == int(a) and atom_order[-1] == int(b):
            return atom_order
        if atom_order[0] == int(b) and atom_order[-1] == int(a):
            return list(reversed(atom_order))

    path = list(Chem.rdmolops.GetShortestPath(mol, int(a), int(b)))
    if len(path) == 4:
        return [int(each) for each in path]
    raise ValueError(f"Cannot find 4pi path between atoms {a} and {b}")

def bond_lengths_for_path(position, path):
    position = np.asarray(position, dtype=float)
    return [
        distance_between(position, path[0], path[1]),
        distance_between(position, path[1], path[2]),
        distance_between(position, path[2], path[3]),
    ]

def bond_trend_record(prefix, lengths, reference_lengths=None):
    record = {
        f"{prefix}_bond12": float(lengths[0]),
        f"{prefix}_bond23": float(lengths[1]),
        f"{prefix}_bond34": float(lengths[2]),
        f"{prefix}_terminal_avg": float((lengths[0] + lengths[2]) / 2.0),
        f"{prefix}_alternation": float(lengths[1] - (lengths[0] + lengths[2]) / 2.0),
        f"{prefix}_asynchronicity": float(abs(lengths[0] - lengths[2])),
    }
    if reference_lengths is not None:
        record.update({
            f"{prefix}_delta_bond12": float(lengths[0] - reference_lengths[0]),
            f"{prefix}_delta_bond23": float(lengths[1] - reference_lengths[1]),
            f"{prefix}_delta_bond34": float(lengths[2] - reference_lengths[2]),
            f"{prefix}_delta_terminal_avg": float(((lengths[0] + lengths[2]) - (reference_lengths[0] + reference_lengths[2])) / 2.0),
            f"{prefix}_delta_alternation": float(
                (lengths[1] - (lengths[0] + lengths[2]) / 2.0)
                - (reference_lengths[1] - (reference_lengths[0] + reference_lengths[2]) / 2.0)
            ),
            f"{prefix}_delta_asynchronicity": float(abs(lengths[0] - lengths[2]) - abs(reference_lengths[0] - reference_lengths[2])),
        })
    return record

def descriptor_lookup(input_csv=INPUT_CSV):
    df = pd.read_csv(input_csv).copy()
    if "Diene_Index" in df.columns:
        df["Diene_Index"] = df["Diene_Index"].map(to_int)
    if "Ene_Index" in df.columns:
        df["Ene_Index"] = df["Ene_Index"].map(to_int)
    if "Structure_id" in df.columns:
        df["Structure_id"] = df["Structure_id"].map(to_int)
    if "conf_id" in df.columns:
        df["conf_id"] = df["conf_id"].map(to_int)
    return df

def matching_descriptor_row(descriptor_df, diene_index, ene_index=None, structure_id=None, conf_id=None):
    subset = descriptor_df[descriptor_df["Diene_Index"].eq(to_int(diene_index))]
    if ene_index is not None and "Ene_Index" in subset.columns:
        subset = subset[subset["Ene_Index"].eq(to_int(ene_index))]
    if structure_id is not None and "Structure_id" in subset.columns:
        exact = subset[subset["Structure_id"].eq(to_int(structure_id))]
        if len(exact):
            subset = exact
    if conf_id is not None and "conf_id" in subset.columns:
        exact = subset[subset["conf_id"].eq(to_int(conf_id))]
        if len(exact):
            subset = exact
    if len(subset) == 0:
        return pd.Series(dtype=object)
    return subset.iloc[0]

def descriptor_fields(row):
    result = {}
    for col in ["Diene_Distort", "Ene_Distort", "Interaction", "deltaGa(Solvent)", "deltaGa", "deltaG", "Diene", "Ene"]:
        if col in row:
            result[col] = row[col]
    return result

def relaxed_diene_geometry(diene_index, stable_table=None):
    if stable_table is None:
        stable_table = load_stable_conf_table(ALL_SMILES_CSV)
    stable_conf_id = get_stable_conf_id(diene_index, stable_table)
    symbols, position, charge, mol, log_path, mol_path = read_diene_structure(diene_index, stable_conf_id)
    return {
        "symbols": symbols,
        "position": position,
        "charge": charge,
        "mol": mol,
        "stable_conf_id": stable_conf_id,
        "log_path": log_path,
        "mol_path": mol_path,
    }

def collect_da_4pi_bond_trends(
    da_ts_dirs=DA_TS_DIRS,
    ene_index=DA_ENE_INDEX_FOR_COMPARISON,
    input_csv=INPUT_CSV,
    analysis_dir=ANALYSIS_DIR,
):
    descriptor_df = descriptor_lookup(input_csv)
    stable_table = load_stable_conf_table(ALL_SMILES_CSV)
    diene_cache = {}
    rows, failures = [], []

    for da_ts_dir in da_ts_dirs:
        da_ts_dir = Path(da_ts_dir)
        for log_path in tqdm(sorted(da_ts_dir.glob("*.log")), desc=f"DA {da_ts_dir.name}"):
            parsed_name = parse_da_ts_filename(log_path)
            if parsed_name is None:
                continue
            if ene_index is not None and parsed_name["ene_index"] != int(ene_index):
                continue
            diene_index = parsed_name["diene_index"]
            try:
                if diene_index not in diene_cache:
                    diene_cache[diene_index] = relaxed_diene_geometry(diene_index, stable_table)
                diene_info = diene_cache[diene_index]

                ts_log = logfile_process.Logfile(str(log_path))
                a, b = [int(each) for each in ts_log.title[:2]]
                path = diene_4pi_path(diene_info["mol"], a, b)
                relaxed_lengths = bond_lengths_for_path(diene_info["position"], path)
                ts_lengths = bond_lengths_for_path(ts_log.running_positions[-1], path)
                desc = matching_descriptor_row(
                    descriptor_df,
                    diene_index=diene_index,
                    ene_index=parsed_name["ene_index"],
                    structure_id=parsed_name["structure_id"],
                    conf_id=parsed_name["conf_id"],
                )

                record = {
                    **parsed_name,
                    "da_ts_log_path": str(log_path),
                    "da_source_dir": str(da_ts_dir),
                    "a": a,
                    "b": b,
                    "diene_4pi_path": " ".join(str(each) for each in path),
                    "stable_conf_id": diene_info["stable_conf_id"],
                    "relaxed_diene_log_path": str(diene_info["log_path"]),
                }
                record.update(descriptor_fields(desc))
                record.update(bond_trend_record("relaxed", relaxed_lengths))
                record.update(bond_trend_record("da_ts", ts_lengths, relaxed_lengths))
                rows.append(record)
            except Exception as exc:
                failures.append({
                    **parsed_name,
                    "da_ts_log_path": str(log_path),
                    "error": repr(exc),
                })

    da_df = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failures)
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    da_df.to_csv(analysis_dir / "DA_18100_4pi_bond_trends_all.csv", index=False)
    if len(failed_df):
        failed_df.to_csv(analysis_dir / "DA_18100_4pi_bond_trends_failed.csv", index=False)

    if len(da_df):
        sort_cols = ["diene_index"]
        if "deltaGa(Solvent)" in da_df.columns:
            sort_cols.append("deltaGa(Solvent)")
        da_best_df = da_df.sort_values(sort_cols).groupby("diene_index", as_index=False).first()
    else:
        da_best_df = pd.DataFrame()
    if len(da_best_df):
        da_best_df.to_csv(analysis_dir / "DA_18100_4pi_bond_trends_best_by_diene.csv", index=False)

    print(f"Collected {len(da_df)} DA TS 4pi bond-trend rows for ene_index={ene_index}")
    if len(failed_df):
        print(f"Failed DA TS rows: {len(failed_df)}")
    return da_df, da_best_df, failed_df

def collect_mes_ts_4pi_bond_trends(
    summary_csv=ANALYSIS_DIR / "MeS_thermo_kinetic_summary.csv",
    analysis_dir=ANALYSIS_DIR,
):
    summary_df = pd.read_csv(summary_csv)
    summary_df = summary_df.dropna(subset=["ts_stem"]).copy()
    stable_table = load_stable_conf_table(ALL_SMILES_CSV)
    diene_cache = {}
    rows, failures = [], []

    for _, item in tqdm(summary_df.iterrows(), total=len(summary_df), desc="MeS TS"):
        diene_index = to_int(item["diene_index"])
        ts_stem = str(item["ts_stem"])
        ts_log_path = Path(str(item.get("ts_success_log_path", "")))
        if not ts_log_path.exists():
            ts_log_path = OUTPUT_DIR / "ts_success" / f"{ts_stem}.log"
        try:
            if diene_index not in diene_cache:
                diene_cache[diene_index] = relaxed_diene_geometry(diene_index, stable_table)
            diene_info = diene_cache[diene_index]

            ts_log = logfile_process.Logfile(str(ts_log_path))
            a, b = [int(each) for each in ts_log.title[:2]]
            path = diene_4pi_path(diene_info["mol"], a, b)
            relaxed_lengths = bond_lengths_for_path(diene_info["position"], path)
            ts_lengths = bond_lengths_for_path(ts_log.running_positions[-1], path)

            record = item.to_dict()
            record.update({
                "diene_index": diene_index,
                "mes_ts_log_path": str(ts_log_path),
                "a": a,
                "b": b,
                "diene_4pi_path": " ".join(str(each) for each in path),
                "stable_conf_id": diene_info["stable_conf_id"],
                "relaxed_diene_log_path": str(diene_info["log_path"]),
            })
            record.update(bond_trend_record("relaxed", relaxed_lengths))
            record.update(bond_trend_record("mes_ts", ts_lengths, relaxed_lengths))
            rows.append(record)
        except Exception as exc:
            failures.append({
                "diene_index": diene_index,
                "ts_stem": ts_stem,
                "mes_ts_log_path": str(ts_log_path),
                "error": repr(exc),
            })

    mes_df = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failures)
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mes_df.to_csv(analysis_dir / "MeS_success_TS_4pi_bond_trends.csv", index=False)
    if len(failed_df):
        failed_df.to_csv(analysis_dir / "MeS_success_TS_4pi_bond_trends_failed.csv", index=False)
    print(f"Collected {len(mes_df)} MeS success TS 4pi bond-trend rows")
    if len(failed_df):
        print(f"Failed MeS TS rows: {len(failed_df)}")
    return mes_df, failed_df

def merge_da_mes_4pi_bond_trends(
    da_best_df,
    mes_df,
    analysis_dir=ANALYSIS_DIR,
):
    da_cols = [
        "diene_index", "da_ts_log_path", "structure_id", "conf_id", "deltaGa(Solvent)",
        "da_ts_delta_bond12", "da_ts_delta_bond23", "da_ts_delta_bond34",
        "da_ts_delta_terminal_avg", "da_ts_delta_alternation", "da_ts_delta_asynchronicity",
    ]
    mes_cols = [
        "diene_index", "ts_stem", "MeS_deltaGa", "Diene_Distort",
        "mes_ts_delta_bond12", "mes_ts_delta_bond23", "mes_ts_delta_bond34",
        "mes_ts_delta_terminal_avg", "mes_ts_delta_alternation", "mes_ts_delta_asynchronicity",
    ]
    merged = da_best_df[[col for col in da_cols if col in da_best_df.columns]].merge(
        mes_df[[col for col in mes_cols if col in mes_df.columns]],
        on="diene_index",
        how="inner",
        suffixes=("_da", "_mes"),
    )

    correlation_cols = [
        "deltaGa(Solvent)", "MeS_deltaGa", "Diene_Distort",
        "da_ts_delta_bond12", "da_ts_delta_bond23", "da_ts_delta_bond34",
        "da_ts_delta_terminal_avg", "da_ts_delta_alternation",
        "mes_ts_delta_bond12", "mes_ts_delta_bond23", "mes_ts_delta_bond34",
        "mes_ts_delta_terminal_avg", "mes_ts_delta_alternation",
    ]
    correlation_cols = [col for col in correlation_cols if col in merged.columns]
    corr = merged[correlation_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")

    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(analysis_dir / "DA_vs_MeS_4pi_bond_trend_merged.csv", index=False)
    corr.to_csv(analysis_dir / "DA_vs_MeS_4pi_bond_trend_pearson.csv")
    return merged, corr

def plot_da_mes_4pi_bond_trends(merged_df):
    from matplotlib import pyplot as plt

    pairs = [
        ("da_ts_delta_terminal_avg", "mes_ts_delta_terminal_avg"),
        ("da_ts_delta_alternation", "mes_ts_delta_alternation"),
        ("Diene_Distort", "MeS_deltaGa"),
        ("da_ts_delta_alternation", "MeS_deltaGa"),
    ]
    pairs = [(x, y) for x, y in pairs if x in merged_df.columns and y in merged_df.columns]
    if not pairs:
        print("No paired columns available for 4pi trend plots.")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.6))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (x_col, y_col) in zip(axes, pairs):
        data = merged_df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(data[x_col], data[y_col], s=30)
        if len(data) >= 2:
            x = data[x_col].to_numpy(dtype=float)
            y = data[y_col].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xs, slope * xs + intercept)
            ax.set_title(f"r={np.corrcoef(x, y)[0, 1]:.2f}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    plt.tight_layout()
    plt.show()
