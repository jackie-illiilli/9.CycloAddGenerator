import csv
import json
import re
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, RDLogger

ROOT = Path(r'E:\work\Secondary_Selection\Diene_Ene_Smiles')
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
AUDIT = ROOT / ('all_smiles_recovery_' + STAMP)
AUDIT.mkdir()
shutil.copy2(ROOT / 'all_smiles.csv', AUDIT / 'all_smiles.before.csv')
RDLogger.DisableLog('rdApp.warning')

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

rows = read_csv(ROOT / 'all_smiles.csv')
assert len({int(r['Index']) for r in rows}) == len(rows)
records = {int(r['Index']): r for r in rows}
original = set(records)
mol_files = {int(p.stem.split('_')[1]): p for p in (ROOT / 'mol').glob('smilesid_*.mol')}
added = []
for idx in sorted(mol_files.keys() - records.keys()):
    mol = Chem.MolFromMolFile(str(mol_files[idx]))
    if mol is None:
        raise ValueError(f'Cannot parse mol: {mol_files[idx]}')
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    records[idx] = {'Index': str(idx), 'Smiles': smiles}
    added.append(idx)
print(f'Restored {len(added)} missing SMILES; total {len(records)}', flush=True)
backup_only = []
for old in read_csv(ROOT / 'all_smiles_backup.csv'):
    idx = int(old['Index'])
    if idx not in records:
        mol = Chem.MolFromSmiles(old['Smiles'])
        assert mol is not None
        records[idx] = {'Index': str(idx), 'Smiles': Chem.MolToSmiles(mol)}
        backup_only.append(idx)

opts = {}
for p in sorted((ROOT / 'mol_dft').glob('smilesid_*.log')):
    m = re.fullmatch(r'smilesid_(\d+)_(\d+)\.log', p.name)
    if m:
        opts.setdefault(int(m[1]), []).append((int(m[2]), p))

num = r'([-+]?\d+\.\d+(?:[DEde][-+]?\d+)?)'
scf_re = re.compile(r'SCF Done:.*?=\s*' + num)
g_re = re.compile(r'Thermal correction to Gibbs Free Energy=\s*' + num)

def parse(path, correction=False):
    text = path.read_text(encoding='utf-8', errors='replace')
    # Match the project's parser: first SCF after the last standard orientation.
    tail = text.rsplit('Standard orientation:', 1)
    if len(tail) != 2:
        raise ValueError('No Standard orientation')
    scf = scf_re.search(tail[1])
    if scf is None:
        raise ValueError('No SCF energy')
    value = float(scf[1].replace('D', 'E'))
    if correction:
        g = g_re.search(tail[1])
        if g is None:
            raise ValueError('No Gibbs correction')
        value = float(g[1].replace('D', 'E'))
    return value, 'Normal termination of Gaussian' in text[-3000:]

def analyze(idx):
    row = dict(records[idx])
    row.update({'G/Hatree': '', 'G(Solvent)/Hatree': '', 'E/Hatree': '', 'Stable_conf_id': -1})
    if idx not in mol_files:
        row['G/Hatree'] = 'RDKIT Generate Fail'
        return row, []
    issues = []
    candidates = {'mol_dft_eng': [], 'mol_dft_eng_solvent': []}
    for conf, opt in opts.get(idx, []):
        try:
            corr, ok = parse(opt, True)
            if not ok:
                issues.append([idx, conf, str(opt), 'Optimization not normally terminated'])
                continue
        except Exception as e:
            issues.append([idx, conf, str(opt), str(e)])
            continue
        for folder in candidates:
            log = ROOT / folder / opt.name
            out = log.with_suffix('.out')
            if not log.exists():
                if out.exists():
                    raise RuntimeError(f'ORCA file requires parser: {out}')
                continue
            try:
                energy, ok = parse(log)
                if not ok:
                    raise ValueError('Single point not normally terminated')
                candidates[folder].append((energy + corr, conf, energy))
            except Exception as e:
                issues.append([idx, conf, str(log), str(e)])
    gas = candidates['mol_dft_eng']
    solv = candidates['mol_dft_eng_solvent']
    if gas:
        g, conf, e = min(gas)
        row.update({'G/Hatree': g, 'E/Hatree': e, 'Stable_conf_id': conf})
        if solv:
            gs, cs, _ = min(solv)
            row.update({'G(Solvent)/Hatree': gs, 'Stable_conf_id': cs})
        else:
            row.update({'G(Solvent)/Hatree': 'DFT ENG Fail', 'Stable_conf_id': -1})
    else:
        row['G/Hatree'] = 'DFT OPT Fail' if idx not in opts else 'DFT ENG Fail'
    return row, issues

final = []
issues = []
with ThreadPoolExecutor(max_workers=8) as pool:
    for i, (row, row_issues) in enumerate(pool.map(analyze, sorted(records)), 1):
        final.append(row)
        issues.extend(row_issues)
        if i % 1000 == 0:
            print(f'Parsed {i}/{len(records)}', flush=True)

columns = ['Index', 'Smiles', 'G/Hatree', 'G(Solvent)/Hatree', 'E/Hatree', 'Stable_conf_id']
tmp = AUDIT / 'all_smiles.recovered.csv'
with tmp.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(final)
with (AUDIT / 'log_issues.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(['Index', 'conf_id', 'file', 'issue']); w.writerows(issues)
loaded = read_csv(tmp)
assert len(loaded) == len(records)
assert {int(r['Index']) for r in loaded} == set(records)
assert all(r['Smiles'] == records[int(r['Index'])]['Smiles'] for r in loaded)
assert all(Chem.MolFromSmiles(r['Smiles']) is not None for r in loaded)
lookup = {int(r['Index']): r for r in final}
comparisons = {}
for name in ['all_smiles_backup.csv', 'all_smiles_appeared.csv']:
    differences = []
    for old in read_csv(ROOT / name):
        idx = int(old['Index'])
        if idx not in lookup:
            differences.append([idx, 'Index', 'missing', old['Index']]); continue
        new = lookup[idx]
        for col in columns[2:]:
            try:
                same = abs(float(new[col]) - float(old[col])) < 2e-7
            except (ValueError, TypeError):
                same = str(new[col]) == old[col]
            if not same:
                differences.append([idx, col, new[col], old[col]])
    comparisons[name] = len(differences)
    with (AUDIT / (name + '.differences.csv')).open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['Index', 'column', 'recovered', 'reference']); w.writerows(differences)
summary = {'original_rows': len(rows), 'added_rows': len(added), 'final_rows': len(final),
           'added_ids': added, 'backup_only_ids': backup_only,
           'missing_mol_ids_in_range': sorted(set(range(max(records)+1)) - set(mol_files)),
           'complete_energy_rows': sum(r['Stable_conf_id'] >= 0 for r in final),
           'incomplete_ids': [int(r['Index']) for r in final if r['Stable_conf_id'] < 0],
           'log_issues': len(issues), 'reference_difference_cells': comparisons}
(AUDIT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('added_ids', 'incomplete_ids')}), flush=True)
print(f'REVIEW CANDIDATE: {tmp}', flush=True)
