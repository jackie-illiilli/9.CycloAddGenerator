"""Draw unique diene/dienophile catalogs as SI PNGs and one vector PDF.

Requires RDKit, reportlab, svglib, pypdf, and Poppler's pdftoppm.
"""

import argparse
import csv
from io import BytesIO
from math import ceil
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.graphics import renderPDF
from reportlab.pdfgen.canvas import Canvas
from svglib.svglib import svg2rlg


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
COLORS = {"Diene": (0.20, 0.55, 0.95), "Ene": (1.00, 0.58, 0.20)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def molecule(smiles):
    result = Chem.MolFromSmiles(smiles)
    if result is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.RemoveHs(result)


def graph_key(mol):
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, isomericSmiles=False)


def collect_reactants():
    rows = read_csv(ROOT / "Data/Predict_DI/5_DI_selected.csv")
    mappings = read_csv(HERE / "SI_figures/figure_manifest.csv")
    if len(rows) != len(mappings):
        raise ValueError("The reaction CSV and mapping manifest differ in length")
    groups = {"Diene": {}, "Ene": {}}
    for position, (row, mapping) in enumerate(zip(rows, mappings), 1):
        for field in ("Diene_Index", "Ene_Index", "Structure_Id"):
            if row[field] != mapping[field]:
                raise ValueError(f"Mapping mismatch at row {position}: {field}")
        if int(mapping["csv_row"]) != position:
            raise ValueError("The mapping manifest is not in CSV order")
        reactant_smiles, _ = mapping["reaction_aam"].split(">>")
        fragments = reactant_smiles.split(".")
        if len(fragments) != 2:
            raise ValueError("Expected exactly two reactant fragments")
        import json
        forming_maps = {number for pair in json.loads(mapping["formed_bonds"]) for number in pair}
        for role, mapped_smiles in zip(("Diene", "Ene"), fragments):
            identifier = int(row[role + "_Index"])
            source = molecule(row[role])
            item = groups[role].setdefault(identifier, {
                "id": identifier, "smiles": row[role], "mol": source,
                "count": 0, "sites": set(), "bonds": set(),
            })
            if item["smiles"] != row[role]:
                raise ValueError(f"Inconsistent SMILES for {role} {identifier}")
            mapped = molecule(mapped_smiles)
            if graph_key(mapped) != graph_key(item["mol"]):
                raise ValueError(f"Mapped structure mismatch for {role} {identifier}")
            query = Chem.Mol(mapped)
            for atom in query.GetAtoms():
                atom.SetAtomMapNum(0)
            match = item["mol"].GetSubstructMatch(query, useChirality=False)
            if len(match) != item["mol"].GetNumAtoms():
                raise ValueError(f"Cannot transfer sites for {role} {identifier}")
            sites = {match[atom.GetIdx()] for atom in mapped.GetAtoms()
                     if atom.GetAtomMapNum() in forming_maps}
            if len(sites) != 2:
                raise ValueError(f"Expected two bond-forming atoms for {role} {identifier}")
            item["count"] += 1
            item["sites"].update(sites)
            if role == "Ene":
                bond = item["mol"].GetBondBetweenAtoms(*sorted(sites))
                if bond is None:
                    raise ValueError("The dienophile reaction atoms must be bonded")
                item["bonds"].add(bond.GetIdx())
    for role, catalog in groups.items():
        if len(catalog) != len({row[role] for row in rows}):
            raise ValueError(f"ID and SMILES deduplication counts differ for {role}")
        if sum(item["count"] for item in catalog.values()) != len(rows):
            raise ValueError("Reactant occurrence counts do not sum to the reaction count")
    total_unique = len({row[role] for row in rows for role in ("Diene", "Ene")})
    return rows, groups, total_unique


def svg_text(parent, x, y, text, size, bold=False, centered=False):
    attributes = {"x": str(x), "y": str(y), "font-size": str(size),
                  "font-family": "Helvetica", "fill": "black"}
    if bold:
        attributes["font-weight"] = "bold"
    if centered:
        attributes["text-anchor"] = "middle"
    element = ET.SubElement(parent, f"{{{NS}}}text", attributes)
    element.text = text


def compound_svg(item, role):
    drawing = Chem.Mol(item["mol"])
    assert all(atom.GetAtomicNum() != 1 for atom in drawing.GetAtoms())
    rdDepictor.Compute2DCoords(drawing)
    drawing = rdMolDraw2D.PrepareMolForDrawing(drawing, addChiralHs=False)
    drawer = rdMolDraw2D.MolDraw2DSVG(440, 235)
    options = drawer.drawOptions()
    options.prepareMolsBeforeDrawing = False
    options.addStereoAnnotation = False
    options.useBWAtomPalette()
    options.flagCloseContactsDist = 0
    options.padding = 0.16
    color = COLORS[role]
    drawer.DrawMolecule(drawing, highlightAtoms=sorted(item["sites"]),
                        highlightBonds=sorted(item["bonds"]),
                        highlightAtomColors={index: color for index in item["sites"]},
                        highlightBondColors={index: color for index in item["bonds"]})
    drawer.FinishDrawing()
    root = ET.fromstring(drawer.GetDrawingText())
    root.tag = f"{{{NS}}}g"
    root.attrib.clear()
    return root


def catalog_svg(items, role, first, count, reaction_count, total_unique, shared_ids):
    columns, cell_width, cell_height = 3, 440, 280
    header, footer = 105, 105
    width = columns * cell_width
    height = header + ceil(len(items) / columns) * cell_height + footer
    root = ET.Element(f"{{{NS}}}svg", {
        "width": "180mm", "height": f"{height / width * 180:.6f}mm",
        "viewBox": f"0 0 {width} {height}",
    })
    ET.SubElement(root, f"{{{NS}}}rect", {"width": str(width), "height": str(height), "fill": "white"})
    title = "Dienes" if role == "Diene" else "Dienophiles (Ene)"
    svg_text(root, 18, 38, f"{title}: {count} unique compounds", 32, True)
    svg_text(root, 18, 76,
             f"{reaction_count} reactions; 51 dienes; 25 dienophiles; {total_unique} unique compounds overall.", 22)
    prefix = "D" if role == "Diene" else "E"
    for index, item in enumerate(items):
        row, column = divmod(index, columns)
        remainder = len(items) % columns
        x_offset = (columns - remainder) * cell_width / 2 if row == len(items) // columns and remainder else 0
        group = compound_svg(item, role)
        group.set("transform", f"translate({column * cell_width + x_offset},{header + row * cell_height})")
        root.append(group)
        svg_text(root, column * cell_width + x_offset + cell_width / 2,
                 header + row * cell_height + 264,
                 f"{prefix}{first + index:02d}   ID {item['id']:05d}   n={item['count']}", 23, True, True)
    svg_text(root, 18, height - 68, "n = occurrences among the 84 selected reactions. Order: first appearance in the CSV.", 19)
    svg_text(root, 18, height - 40,
             "Colored atoms mark mapped reactive sites; symmetry-equivalent assignments may differ.", 19)
    shared = ", ".join(f"{identifier:05d}" for identifier in shared_ids)
    svg_text(root, 18, height - 12, f"Compound {shared} is listed in both roles. Explicit H atoms and R/S labels are omitted.", 19)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdftoppm", default=shutil.which("pdftoppm"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if not args.pdftoppm:
        parser.error("Provide the path to Poppler using --pdftoppm")
    rows, groups, total_unique = collect_reactants()
    shared_ids = sorted(groups["Diene"].keys() & groups["Ene"].keys())
    output = HERE / "SI_reactants"
    output.mkdir(exist_ok=True)
    pdf_path = output / "SI_reactants_all.pdf"
    canvas = Canvas(str(pdf_path))
    canvas.setTitle("Unique reactants in the 84 selected cycloaddition reactions")
    pages = []
    for role, per_page in (("Diene", 18), ("Ene", 15)):
        items = list(groups[role].values())
        for start in range(0, len(items), per_page):
            subset = items[start:start + per_page]
            data = catalog_svg(subset, role, start + 1, len(items), len(rows), total_unique, shared_ids)
            drawing = svg2rlg(BytesIO(data))
            if drawing is None:
                raise ValueError("Cannot convert the reactant catalog to PDF")
            canvas.setPageSize((drawing.width, drawing.height))
            renderPDF.draw(drawing, canvas, 0, 0)
            canvas.showPage()
            pages.append((f"SI_{role.lower()}s_{start // per_page + 1:02d}", len(subset)))
    canvas.save()
    for page, (name, count) in enumerate(pages, 1):
        subprocess.run([args.pdftoppm, "-f", str(page), "-l", str(page), "-singlefile",
                        "-png", "-r", str(args.dpi), str(pdf_path), str(output / name)], check=True)
        print(f"Created {name}.png: {count} compounds", flush=True)
    print(f"{len(rows)} reactions: {len(groups['Diene'])} dienes, {len(groups['Ene'])} dienophiles, "
          f"{total_unique} unique compounds overall; shared IDs: {shared_ids}")
    print(f"Saved {len(pages)} PNGs and one vector PDF to {output}")


if __name__ == "__main__":
    main()
