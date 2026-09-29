"""Paths shared by discovery and MeS notebooks. No scientific jobs run on import.

Set environment variables before importing these modules; restart the notebook
kernel after changing them, because function defaults bind these paths at import.
"""
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
SECONDARY_ROOT = Path(os.environ.get("CYCLOADD_SECONDARY_ROOT", r"E:\work\Secondary_Selection"))
SMILES_ROOT = SECONDARY_ROOT / "Diene_Ene_Smiles"
PRED_DIR = SECONDARY_ROOT / "All_possible_DA_ZINC"
ONE_PATH_DIR = SECONDARY_ROOT / "All_possible_DA_One_ZINC"
DESCRIPTOR_MAP = Path(os.environ.get("CYCLOADD_DESCRIPTOR_MAP", str(SMILES_ROOT / "all_property_detailSterimol_witharea.pkl")))
DFT_DIR = REPO_ROOT / "Data/DFT_Result"
SCREEN_DIR = REPO_ROOT / "Data/Predict_DI"
FMO_CSV = Path(os.environ.get("CYCLOADD_FMO_CSV", str(REPO_ROOT / "Data/FMO_Energy.csv")))
MES_OUTPUT_DIR = Path(os.environ.get("CYCLOADD_MES_OUTPUT_DIR", str(SECONDARY_ROOT / "7_MAAtest_MeS_attack_gjf")))
_mes_candidates = [SCREEN_DIR / "7_MAAtest_DA.csv", SCREEN_DIR / "4p-2p/7_MAAtest_DA.csv"]
MES_INPUT_CSV = Path(os.environ.get("CYCLOADD_MES_INPUT_CSV", str(next((p for p in _mes_candidates if p.exists()), _mes_candidates[0]))))
PURCHASE_LEDGER = Path(os.environ.get("CYCLOADD_PURCHASE_LEDGER", str(SCREEN_DIR / "5_DI_selected_Index_smiles.csv")))

def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path
