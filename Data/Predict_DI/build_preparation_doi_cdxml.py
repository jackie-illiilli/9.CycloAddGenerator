"""Create a ChemDraw catalog of Reaxys-reported compounds with preparation DOIs.

Read the DOI from its named column or the current unnamed final CSV column.
Retain recorded DOI identifiers without requesting or inferring paper contents.
Requires RDKit and NumPy; reuses the repository's Exa-style CDXML writer.
"""

import csv
import importlib.util
from math import ceil
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse

import numpy as np
from rdkit import Chem


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Purchasable_smiles.csv"
OUTPUT = HERE / "Reaxys_Preparation_DOI_Molecules.cdxml"
WRITER = HERE / "5_DI_selected_reaction_drawings_with_preparations/build_chemdraw_cdxml.py"


def load_writer():
    spec = importlib.util.spec_from_file_location("reaction_cdxml_writer", WRITER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_doi(value):
    value = (value or "").strip()
    if re.match(r"^https?://", value, re.IGNORECASE):
        parsed = urlparse(value)
        if parsed.netloc.lower() not in {"doi.org", "dx.doi.org", "www.doi.org"}:
            return None
        value = unquote(parsed.path.lstrip("/"))
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value, re.IGNORECASE) else None


def select_compounds():
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        doi_column = next((name for name in fields if name.strip().lower() == "doi"), None)
        if doi_column is None:
            if "" not in fields:
                raise ValueError("No DOI column or unnamed reference column was found")
            doi_column = ""
        rows = list(reader)
    identifiers = [int(row["Index"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("The source CSV contains duplicate compound IDs")
    selected, excluded = [], []
    for row in rows:
        if not all((row.get(field) or "").strip().upper() == "TRUE"
                   for field in ("In Reaxys", "Preparations")):
            continue
        doi = normalize_doi(row.get(doi_column))
        if doi is None:
            excluded.append((row["Index"], row.get(doi_column) or ""))
        else:
            selected.append({**row, "doi": doi})
    if not selected:
        raise ValueError("No Reaxys-reported compounds have a recorded preparation DOI")
    return selected, excluded


def build_catalog(records, writer):
    document = writer.Document()
    document.root.set("Name", "Reaxys-reported molecules with preparation DOIs")
    document.root.set("Comment", "Preparation DOI identifiers recorded in Purchasable_smiles.csv")
    columns, rows_per_sheet = 4, 5
    per_sheet = columns * rows_per_sheet
    sheet_count = ceil(len(records) / per_sheet)
    page = document.element(document.root, "page", WidthPages="1", HeightPages=str(sheet_count),
                            HeaderPosition="36", FooterPosition="36",
                            BoundingBox=writer.coordinates((0, 0, writer.PAGE_WIDTH, sheet_count * writer.PAGE_HEIGHT)))
    expected = []
    for sheet_index in range(sheet_count):
        document.offset = np.asarray([0, sheet_index * writer.PAGE_HEIGHT])
        first = sheet_index * per_sheet
        batch = records[first:first + per_sheet]
        document.text(page, 6.52, 17, "Reaxys-reported compounds with preparation DOIs", 11, True)
        document.text(page, 6.52, 37, f"Compounds {first + 1}-{first + len(batch)} of {len(records)}", 9)
        column_width = writer.PAGE_WIDTH / columns
        for position, record in enumerate(batch):
            row, column = divmod(position, columns)
            # Center the incomplete final row without changing the molecule size.
            remainder = len(batch) % columns
            x_offset = ((columns - remainder) * column_width / 2
                        if row == len(batch) // columns and remainder else 0)
            x = x_offset + (column + 0.5) * column_width
            top = 57 + row * 134
            mol = writer.molecule(record["Smiles"])
            expected.append(mol)
            document.fragment(page, mol, (x, top + 43), {}, {})
            identifier = document.text(page, x, top + 90, f"ID {int(record['Index']):05d}", 8.5, True)
            identifier.set("Justification", "Center")
            reference = document.text(page, x, top + 107, record["doi"], 8)
            reference.set("Justification", "Center")
        document.text(page, 6.52, writer.PAGE_HEIGHT - 16,
                      "Structures and preparation DOI identifiers follow Purchasable_smiles.csv.", 8)
    document.save(OUTPUT)
    writer.validate(OUTPUT, expected)
    return document, sheet_count


def validate_catalog(document, records, writer, sheet_count):
    root = ET.parse(OUTPUT).getroot()
    pages = root.findall("page")
    if len(pages) != 1 or int(pages[0].get("HeightPages")) != sheet_count:
        raise ValueError("The catalog must use one drawing space with separate vertical sheets")
    fragments = root.findall(".//fragment")
    if len(fragments) != len(records):
        raise ValueError("Molecule count does not match the qualifying source rows")
    references = [text for text in root.findall(".//t")
                  if (value := "".join(text.itertext())).strip().startswith("10.")]
    if ["".join(text.itertext()).strip() for text in references] != [r["doi"] for r in records]:
        raise ValueError("DOI labels differ from the selected source records")
    for index, (fragment, reference) in enumerate(zip(fragments, references)):
        left, top, right, bottom = map(float, fragment.get("BoundingBox").split())
        sheet = index // 20
        x, y = map(float, reference.get("p").split())
        if not (0 <= left <= right <= writer.PAGE_WIDTH
                and sheet * writer.PAGE_HEIGHT <= top <= bottom < (sheet + 1) * writer.PAGE_HEIGHT
                and y > bottom):
            raise ValueError(f"Structure or DOI label is outside its page/cell at index {index + 1}")
        if reference.get("Justification") != "Center":
            raise ValueError("DOI labels must be centered below the structures")
    # The catalog uses black chemical objects and no reaction-site graphics.
    if any(node.get("color") != "0" for node in root.findall(".//n") + root.findall(".//b")):
        raise ValueError("The molecule catalog must use black atoms and bonds")


def main():
    records, excluded = select_compounds()
    writer = load_writer()
    document, sheets = build_catalog(records, writer)
    validate_catalog(document, records, writer, sheets)
    print(f"Saved {len(records)} editable structures and their preparation DOIs on {sheets} A4 sheets: {OUTPUT}")
    for identifier, reference in excluded:
        print(f"Excluded compound {identifier}: no DOI identifier in recorded reference ({reference})")


if __name__ == "__main__":
    main()
