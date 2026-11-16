"""Audit repository molecule references and build a source-consistent subset.

Run directly (not through Code.__init__) in the Python 3.8 main environment
to keep NumPy scalar pickles readable by both project environments.
"""
import ast
import csv
import hashlib
import io
import json
import pickle
import re
import warnings
import zipfile
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "Ignored/Descriptors_.pkl"
OLD = ROOT / "Data/DFT_Result/Descriptors_Filter.pkl"
OUTPUT = ROOT / "Data/DFT_Result/Descriptors_Filter_merged.pkl"
AUDIT = ROOT / "Data/DFT_Result/descriptor_audit"
CATALOGS = {
    "Data/ZINC_Data/all_smiles.csv", "Data/FMO_Energy.csv", "Data/Orbit.csv",
    "Data/Analysis_DI_Selection_Reported_BO/FMO_Energy.csv",
    "Data/ZINC_Data/all_smiles.smi", "Data/ZINC_Data/all_diene.smi",
    "Data/ZINC_Data/all_ene.smi",
}
FIELDS = {"diene_index", "ene_index", "diene_indexs", "ene_indexs", "smiles_id"}
CONSTANT_NAMES = {
    "dienes", "enes", "banned_bo_diene", "banned_bo_ene", "banned_diene_id",
    "self_reaction_ids", "da_ene_index_for_comparison", "diene_index", "ene_index",
    "diene_indexs", "ene_indexs", "diene_ids", "ene_ids",
}
KEY = re.compile(r"^(\d+) ([01]) (\d+) (\d+)$")
SMILES_ID = re.compile(r"smilesid_(\d{5})(?!\d)", re.I)
REACTION_FILE = re.compile(r"(?<!\d)(\d{5})_(\d{5})_\d{5}_\d{4}(?!\d)")
LABEL = re.compile(r"\b(?:Diene|Ene|ID)\s*[:=]?\s+(\d{4,5})(?!\d)", re.I)


class DescriptorUnpickler(pickle.Unpickler):
    """Read NumPy 2 scalar namespaces in the project's NumPy 1.x environment."""
    def find_class(self, module, name):
        if int(np.__version__.split(".")[0]) < 2 and module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value):
    text = str(value).strip()
    return int(text.split(".")[0]) if re.fullmatch(r"\d+(?:\.0+)?", text) else None


def normalized(field):
    return re.sub(r"\.\d+$", "", str(field).strip().lower())


def id_columns(headers):
    names = [normalized(h) for h in headers]
    columns = [i for i, name in enumerate(names) if name in FIELDS]
    # 'index' alone can be a reaction row number. Accept it only in a
    # molecule table with a SMILES column and without reactant-index columns.
    if not columns and "smiles" in names and "index" in names:
        columns.append(names.index("index"))
    return columns


def text_ids(text):
    ids = {int(m.group(1)) for m in SMILES_ID.finditer(text)}
    for match in REACTION_FILE.finditer(text):
        ids.update(map(int, match.groups()))
    ids.update(int(m.group(1)) for m in LABEL.finditer(text))
    return ids


class Tables(HTMLParser):
    """Extract explicitly headed columns from saved notebook HTML tables."""
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table": self.rows = []
        if tag == "tr" and self.rows is not None: self.row = []
        if tag in ("th", "td") and self.row is not None: self.cell = []

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append("".join(self.cell).strip()); self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row); self.row = None
        if tag == "table" and self.rows is not None:
            self.tables.append(self.rows); self.rows = None


def table_ids(rows):
    ids = set()
    if not rows: return ids
    columns = id_columns(rows[0])
    for row in rows[1:]:
        for i in columns:
            value = integer(row[i]) if i < len(row) else None
            if value is not None: ids.add(value)
    return ids


def code_ids(source):
    ids = text_ids(source)
    source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                       for line in source.splitlines())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except SyntaxError:
        return ids, False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id.lower() in CONSTANT_NAMES for t in targets):
                try: values = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError): continue
                if not isinstance(values, (list, tuple, set)): values = [values]
                ids.update(v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            name = owner.id.lower() if isinstance(owner, ast.Name) else ""
            if name in CONSTANT_NAMES and node.func.attr == "append" and node.args:
                try: value = ast.literal_eval(node.args[0])
                except (ValueError, TypeError): continue
                if isinstance(value, int) and value >= 0: ids.add(value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = KEY.fullmatch(node.value)
            if match: ids.add(int(match.group(1)))
    return ids, True


def read_csv(path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try: return list(csv.reader(io.StringIO(raw.decode(encoding)))), encoding
        except UnicodeDecodeError: pass
    raise ValueError("Cannot decode " + str(path))


def xlsx_strings(path):
    """Read cell values only; no Excel authoring or formula evaluation."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(n.itertext()) for n in tree.findall("s:si", ns)]
        for name in archive.namelist():
            if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name): continue
            tree = ET.fromstring(archive.read(name))
            for cell in tree.findall(".//s:c", ns):
                value = cell.find("s:v", ns)
                if cell.get("t") == "s" and value is not None:
                    yield strings[int(value.text)]
                elif cell.get("t") == "inlineStr":
                    yield "".join(cell.find("s:is", ns).itertext())
                elif value is not None: yield value.text


def main():
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError("Build with the Python 3.8 main environment (NumPy 1.x) for portable scalar pickles.")
    with FULL.open("rb") as handle: full = DescriptorUnpickler(handle).load()
    with OLD.open("rb") as handle: old = DescriptorUnpickler(handle).load()
    before_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (FULL, OLD)}
    assert all(KEY.fullmatch(k) for k in full.keys() | old.keys())
    full_ids = {int(k.split()[0]) for k in full}
    old_ids = {int(k.split()[0]) for k in old}
    sources, inventory, errors, syntax_skips = {}, [], [], []
    required_sites = defaultdict(set)
    unparsed_site_rows = defaultdict(list)
    smiles_lookup = defaultdict(set)
    catalog_rows, _ = read_csv(ROOT / "Data/ZINC_Data/all_smiles.csv")
    for row in catalog_rows[1:]:
        smiles_lookup[row[1]].add(int(row[0]))

    def record(label, ids, category="used"):
        if ids: sources[label] = {"category": category, "ids": sorted(ids)}

    paths = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not any(part in {".git", "__pycache__", "tmp", ".codex_review"} for part in p.relative_to(ROOT).parts)
                   and AUDIT not in p.parents and p not in {OUTPUT, Path(__file__).resolve()})
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        category = "catalog" if rel in CATALOGS else "used"
        entry = {"path": rel, "bytes": path.stat().st_size}
        record(rel + "#filename", text_ids(path.name))
        try:
            if path.suffix == ".csv":
                rows, encoding = read_csv(path)
                record(rel + "#columns", table_ids(rows), category)
                entry.update(status=category + "_table", rows=max(0, len(rows)-1), encoding=encoding)
                headers = rows[0] if rows else []
                # Independently audit exact site keys used by Code.model_train.read_df.
                if all(field in headers for field in ("Diene_Index", "Ene_Index", "Title")):
                    for row_number, row in enumerate(rows[1:], 2):
                        fields = dict(zip(headers, row))
                        title = fields.get("Title", "").split()
                        if len(title) >= 4 and all(integer(v) is not None for v in title[:4]):
                            d, e = integer(fields.get("Diene_Index")), integer(fields.get("Ene_Index"))
                            if d is not None and e is not None:
                                required_sites[rel].update(("%05d 1 %d %d" % (d, int(title[0]), int(title[1])),
                                                           "%05d 0 %d %d" % (e, int(title[2]), int(title[3]))))
                            else: unparsed_site_rows[rel].append(row_number)
                        else: unparsed_site_rows[rel].append(row_number)
                if not id_columns(headers) and rel == "Data/ZINC_Data/all_selected_smiles.csv":
                    record(rel + "#exact_smiles", set().union(*(smiles_lookup.get(v, set()) for row in rows[1:] for v in row)))
            elif path.suffix in (".pkl", ".pickle"):
                if path == FULL:
                    entry["status"] = "full_descriptor_pool_not_selection_evidence"
                else:
                    with path.open("rb") as handle: mapping = DescriptorUnpickler(handle).load()
                    ids = {int(k.split()[0]) for k in mapping if isinstance(k, str) and KEY.fullmatch(k)}
                    record(rel + "#descriptor_keys", ids)
                    entry["status"] = "descriptor_key_inventory"
            elif path.suffix == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8"))
                for i, cell in enumerate(notebook["cells"]):
                    label = rel + "#cell_%d" % i
                    source = "".join(cell["source"])
                    if cell["cell_type"] == "code":
                        ids, parsed = code_ids(source)
                        if not parsed: syntax_skips.append(label)
                    else: ids = text_ids(source)
                    record(label + "_source", ids)
                    for j, output in enumerate(cell.get("outputs", [])):
                        data = output.get("data", {})
                        html = "".join(data.get("text/html", []))
                        parser = Tables(); parser.feed(html)
                        output_ids = set().union(*(table_ids(rows) for rows in parser.tables)) if parser.tables else set()
                        plain = "".join(output.get("text", [])) + "".join(data.get("text/plain", []))
                        output_ids.update(text_ids(plain))
                        record(label + "_output_%d" % j, output_ids)
                entry["status"] = "notebook_source_and_text_tables"
            elif path.suffix == ".py":
                ids, parsed = code_ids(path.read_text(encoding="utf-8-sig"))
                record(rel + "#source", ids)
                if not parsed: syntax_skips.append(rel)
                entry["status"] = "python_source"
            elif path.suffix == ".xlsx":
                values = list(xlsx_strings(path))
                record(rel + "#exact_smiles", set().union(*(smiles_lookup.get(v, set()) for v in values)))
                record(rel + "#labels", text_ids("\n".join(values)))
                entry["status"] = "xlsx_cell_values"
            elif path.suffix == ".smi":
                values = path.read_text(encoding="utf-8").splitlines()
                record(rel + "#exact_smiles", set().union(*(smiles_lookup.get(v.strip(), set()) for v in values)), category)
                entry["status"] = category + "_smiles"
            elif path.suffix in (".md", ".txt", ".json", ".svg", ".cdxml", ".log", ".xyz", ".mol", ".sh", ""):
                record(rel + "#labels", text_ids(path.read_text(encoding="utf-8", errors="replace")))
                entry["status"] = "text_labels_and_structure_identifiers"
            else:
                entry["status"] = "binary_visual_or_editor_file_no_ocr"
        except Exception as exc:
            entry["status"] = "read_error"
            errors.append({"path": rel, "error": str(exc)})
        inventory.append(entry)
    if errors:
        raise RuntimeError("Incomplete scan; refusing to build: " + json.dumps(errors))
    used = set().union(*(set(item["ids"]) for item in sources.values() if item["category"] == "used"))
    catalogs = set().union(*(set(item["ids"]) for item in sources.values() if item["category"] == "catalog"))
    selected_ids = used | old_ids
    selected = {k: v for k, v in full.items() if int(k.split()[0]) in selected_ids}
    old_only = old.keys() - full.keys()
    selected.update((k, old[k]) for k in sorted(old_only))
    assert old.keys() <= selected.keys()
    assert all(len(v) == (36 if k.split()[1] == "1" else 28) for k, v in selected.items())

    conflicts = []
    for key in sorted(old.keys() & full.keys()):
        a, b = np.asarray(full[key]), np.asarray(old[key])
        positions = np.flatnonzero(~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).tolist()
        if positions:
            conflicts.append({"key": key, "positions_zero_based": positions,
                              "base_features_differ": any(i < len(a)-8 for i in positions),
                              "full_values": [float(a[i]) for i in positions],
                              "old_values": [float(b[i]) for i in positions]})
    # Protocol 4 and NumPy 1.x serialization preserve both Python 3.8/3.12 readers.
    payload = pickle.dumps(selected, protocol=4)
    restored = pickle.loads(payload)
    assert restored.keys() == selected.keys()
    assert all(np.array_equal(np.asarray(restored[k]), np.asarray(v), equal_nan=True) for k, v in selected.items())
    assert all(sha256(ROOT / p) == digest for p, digest in before_hashes.items())
    counts = Counter(int(k.split()[0]) for k in selected)
    by_index = defaultdict(list)
    for source, item in sources.items():
        if item["category"] == "used":
            for idx in item["ids"]: by_index[idx].append(source)
    summary = {
        "policy": "Actual datasets, candidate lists, explicit source/output references and prior descriptor subsets; exclude catalog-only molecules; full source wins for shared keys; retain every site and role for selected indices.",
        "source_sha256": before_hashes,
        "full": {"molecules": len(full_ids), "records": len(full), "bytes": FULL.stat().st_size},
        "old": {"molecules": len(old_ids), "records": len(old), "bytes": OLD.stat().st_size},
        "merged": {"molecules": len(counts), "records": len(selected), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                   "added_molecules": len(set(counts)-old_ids), "added_records": len(selected.keys()-old.keys()),
                   "records_by_role": dict(Counter(k.split()[1] for k in selected))},
        "requested_molecules": len(selected_ids), "missing_indices": sorted(selected_ids-set(counts)),
        "catalog_only_molecules_excluded": len((catalogs & full_ids)-selected_ids),
        "all_catalogs_plus_used_available_molecules": len((catalogs | selected_ids) & full_ids),
        "overlap_records": len(old.keys() & full.keys()), "numeric_conflicts": len(conflicts),
        "base_feature_conflict_keys": [c["key"] for c in conflicts if c["base_features_differ"]],
        "old_only_keys_preserved": sorted(old_only), "files_inventoried": len(inventory),
        "syntax_skips": syntax_skips, "read_errors": errors,
        "unparsed_site_csv_rows": dict(unparsed_site_rows),
        "limitations": ["PDF/raster/editor binary contents were not OCR-scanned; their filenames were checked and the available source tables and notebook text were scanned.",
                        "Only semantically labeled molecule IDs and exact catalog SMILES matches are interpreted; atom, row, reaction and conformation indices are not molecule IDs.",
                        "External paths are not followed; missing source descriptors are reported, not generated."],
        "site_coverage": {p: {"required": len(keys), "present": len(keys & selected.keys()),
                               "missing_keys": sorted(keys-selected.keys())} for p, keys in sorted(required_sites.items())},
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    for name, content in {
        "summary.json": summary, "sources.json": sources, "inventory.json": inventory,
        "conflicts.json": conflicts,
        "molecules.json": [{"Index": idx, "descriptor_records": counts[idx], "in_old_filter": idx in old_ids,
                            "sources": by_index[idx]} for idx in sorted(selected_ids)],
    }.items():
        (AUDIT / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, indices in (("requested_indices.txt", selected_ids), ("included_indices.txt", set(counts)),
                          ("missing_indices.txt", selected_ids-set(counts))):
        (AUDIT / name).write_text("".join("%05d\n" % idx for idx in sorted(indices)), encoding="utf-8")
    OUTPUT.write_bytes(payload)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"site_coverage", "missing_indices", "limitations"}}, indent=2))
    print("Missing molecule descriptors:", len(summary["missing_indices"]))
    print("Written", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
