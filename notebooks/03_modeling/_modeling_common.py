"""Shared data alignment and evaluation helpers for the modeling notebooks.

This module does not train models or read data on import. Call its functions from
the numbered notebooks after setting their input paths explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COLUMNS = {
    "Diene distortion": "Diene_Distort",
    "Ene distortion": "Ene_Distort",
    "Interaction": "Interaction",
    "Aqueous activation free energy": "deltaGa(Solvent)",
}
PHYSORG_FEATURE_COUNT = 48
CATBOOST_PARAMS = dict(task_type="CPU", iterations=10_000,
                       learning_rate=0.01, depth=6, verbose=False)


def load_aligned(csv_path, descriptor_map):
    """Keep target rows in exactly the order returned by ``read_df``."""
    from Code import model_train

    raw = pd.read_csv(csv_path).reset_index(drop=True)
    matrix, retained = model_train.read_df(raw, str(descriptor_map))
    retained = np.asarray(retained, dtype=int)
    frame = raw.iloc[retained].reset_index(drop=True)
    if len(frame) != len(matrix):
        raise RuntimeError("Descriptor and target rows are misaligned.")
    if matrix.ndim != 2 or matrix.shape[1] < PHYSORG_FEATURE_COUNT:
        raise ValueError("The descriptor map lacks the expected PhysOrg columns.")
    return frame, matrix


def load_modeling_data(data_dir, descriptor_map):
    data_ga, x_ga = load_aligned(data_dir / "Active_Energy.csv", descriptor_map)
    data_g, x_g = load_aligned(data_dir / "Rxn.csv", descriptor_map)
    required = {"Diene_Index", "Ene_Index", "Structure_id", "deltaG",
                *TARGET_COLUMNS.values()}
    missing = required.difference(data_ga.columns)
    if missing:
        raise KeyError(f"Active_Energy.csv lacks columns: {sorted(missing)}")
    if "Interaction" not in data_g.columns:
        raise KeyError("Rxn.csv lacks the Interaction column.")
    return data_ga, x_ga, data_g, x_g


def reaction_keys(frame):
    return list(zip(frame["Diene_Index"].astype(int),
                    frame["Ene_Index"].astype(int),
                    frame["Structure_id"].astype(int)))


def broad_training_ids(data_g, data_ga, held_out_ids):
    """Exclude held-out reactions that have D/I labels from broad ΔGrxn training."""
    held_out = set(reaction_keys(data_ga.iloc[held_out_ids]))
    keep = data_g["Interaction"].isna().to_numpy() | np.fromiter(
        (key not in held_out for key in reaction_keys(data_g)),
        dtype=bool, count=len(data_g),
    )
    ids = np.flatnonzero(keep)
    if not len(ids):
        raise ValueError("No broad ΔGrxn training rows remain.")
    return ids


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def new_catboost(seed=0):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(random_seed=seed, **CATBOOST_PARAMS)


def fit_delta_g_feature(data_g, x_g, data_ga, x_ga, train_ids,
                        test_ids, source="broad", seed=0):
    """Fit the first-stage reaction-energy model for one held-out split."""
    model = new_catboost(seed)
    if source == "broad":
        g_ids = broad_training_ids(data_g, data_ga, test_ids)
        model.fit(x_g[g_ids], data_g["deltaG"].to_numpy()[g_ids])
    elif source == "active":
        model.fit(x_ga[train_ids], data_ga["deltaG"].to_numpy()[train_ids])
    else:
        raise ValueError(f"Unknown first-stage source: {source}")
    return model.predict(x_ga)


def evaluate_targets(data_ga, x_train_all, train_ids, test_ids, seed=0):
    rows, predictions = [], {}
    for label, column in TARGET_COLUMNS.items():
        y = data_ga[column].to_numpy()
        model = new_catboost(seed)
        model.fit(x_train_all[train_ids], y[train_ids])
        pred = model.predict(x_train_all[test_ids])
        rows.append({"target": label, **metrics(y[test_ids], pred)})
        predictions[label] = (np.asarray(test_ids), pred)
    return rows, predictions
