"""Export final Gaussian TS geometries in the exact order of an energy CSV."""

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile


SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn "
           "Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
           "In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
           "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu "
           "Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv "
           "Ts Og").split()
ROOT = Path(__file__).resolve().parents[1]


def final_geometry(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.rfind("Normal termination of Gaussian") < max(
        text.rfind("Error termination"), text.rfind("Standard orientation:")
    ):
        raise ValueError(f"No final normal termination: {path}")
    start = text.rfind("Standard orientation:")
    if start < 0 or "Optimization completed" not in text:
        raise ValueError(f"No optimized standard orientation: {path}")
    atoms = []
    for line in text[start:].splitlines()[5:]:
        if line.strip().startswith("---"):
            break
        fields = line.split()
        if len(fields) != 6 or int(fields[0]) != len(atoms) + 1:
            raise ValueError(f"Malformed orientation: {path}")
        number = int(fields[1])
        xyz = tuple(float(value) for value in fields[3:])
        if not 0 < number < len(SYMBOLS) or not all(map(math.isfinite, xyz)):
            raise ValueError(f"Invalid atom or coordinate: {path}")
        atoms.append((SYMBOLS[number], *xyz))
    if not atoms:
        raise ValueError(f"Empty orientation: {path}")
    return atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "Data/DFT_Result/Active_Energy_for_AttFP.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Data/DFT_Result/all_ts.xyz")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("CSV has no data rows")
    identifiers = [
        f"{int(row['Diene_Index']):05d}_{int(row['Ene_Index']):05d}_"
        f"{int(row['Structure_id']):05d}_{int(row['conf_id']):04d}"
        for row in rows
    ]
    missing = [name for name in identifiers if not (args.logs / (name + ".log")).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing logs; first: {missing[:5]}")
    # Replace the destination only after every CSV row has been exported successfully.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=args.output.parent, delete=False) as output:
            temporary = Path(output.name)
            total_atoms = 0
            for index, name in enumerate(identifiers, 1):
                atoms = final_geometry(args.logs / (name + ".log"))
                output.write(f"{len(atoms)}\n{name}\n")
                for symbol, x, y, z in atoms:
                    output.write(f"{symbol} {x} {y} {z}\n")
                total_atoms += len(atoms)
                if index % 500 == 0:
                    print(f"Exported {index}/{len(rows)} frames", flush=True)
        os.replace(temporary, args.output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Wrote {len(rows)} frames ({total_atoms} atoms) to {args.output}")


if __name__ == "__main__":
    main()
