"""
Entraînement et évaluation des 3 modèles ML du projet RAM Delay Intelligence.

Modèle 1 - Classification binaire : un vol sera-t-il à l'heure ou en retard ?
Modèle 2 - Régression           : si retard, quelle durée probable (minutes) ?
Modèle 3 - Classification multi-classe : quel motif (famille) le plus probable,
           une fois le retard constaté (outil d'aide à la saisie) ?

Chaque modèle compare plusieurs algorithmes et conserve le meilleur,
sérialisé avec son préprocesseur dans un unique pipeline sklearn.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

from features import (
    get_model1_data, get_model2_data, get_model3_data,
    CATEGORICAL_FEATURES_PRE, NUMERIC_FEATURES_PRE,
    CATEGORICAL_FEATURES_M3, NUMERIC_FEATURES_M3,
)

RANDOM_STATE = 42
_ROOT = Path(__file__).resolve().parent.parent


def _build_preprocessor(categorical_cols, numeric_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )


def _save_pipeline(pipe: Pipeline, out_path: str, extra: Optional[dict] = None):
    """Sauvegarde un pipeline sklearn de façon robuste aux changements de
    version de bibliothèque.

    Le pickle/joblib direct d'un objet XGBoost (XGBClassifier/XGBRegressor)
    est fragile entre versions majeures de xgboost : le format binaire
    interne du booster ('flux d'entrée corrompu' lors du chargement) peut
    changer. On sépare donc systématiquement :
      - le préprocesseur sklearn (OneHotEncoder/StandardScaler) -> joblib,
        stable car ce sont des objets sklearn purs sans état binaire C++.
      - le dernier step du pipeline ('clf'/'reg') :
          - si c'est un modèle XGBoost -> sérialisé via son format natif
            JSON (`booster.save_model`), garanti rétro-compatible par
            XGBoost lui-même, peu importe la version utilisée ensuite.
          - sinon (LogisticRegression, RandomForest...) -> joblib direct,
            ces objets sklearn n'ont pas ce problème de portabilité.
    """
    final_step_name, final_estimator = pipe.steps[-1]
    is_xgboost = type(final_estimator).__module__.startswith("xgboost")

    payload = {
        "preprocessor": pipe.named_steps["pre"],
        "final_step_name": final_step_name,
        "is_xgboost": is_xgboost,
        "xgboost_kind": None,
        "extra": extra or {},
    }

    out_path = Path(out_path)
    if is_xgboost:
        from xgboost import XGBClassifier, XGBRegressor

        xgb_kind = "classifier" if isinstance(final_estimator, XGBClassifier) else "regressor"
        payload["xgboost_kind"] = xgb_kind
        xgb_json_path = out_path.with_suffix(".xgb.json")
        final_estimator.save_model(str(xgb_json_path))
        payload["xgb_model_path"] = xgb_json_path.name
    else:
        payload["final_estimator"] = final_estimator

    joblib.dump(payload, out_path)


def load_pipeline(path: str):
    """Recharge un pipeline sauvegardé par `_save_pipeline`, en
    reconstruisant un objet scikit-learn Pipeline standard (même
    interface .predict / .predict_proba qu'avant)."""
    path = Path(path)
    payload = joblib.load(path)

    if payload["is_xgboost"]:
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if payload["xgboost_kind"] == "classifier" else XGBRegressor
        final_estimator = cls()
        xgb_json_path = path.parent / payload["xgb_model_path"]
        final_estimator.load_model(str(xgb_json_path))
    else:
        final_estimator = payload["final_estimator"]

    pipe = Pipeline([("pre", payload["preprocessor"]), (payload["final_step_name"], final_estimator)])
    return pipe, payload["extra"]


# ----------------------------------------------------------------------
# Modèle 1 : classification binaire (à l'heure / en retard)
# ----------------------------------------------------------------------
def train_model1(df: pd.DataFrame, out_path: str = None):
    out_path = out_path or str(_ROOT / "models" / "model1_binary.joblib")
    X, y = get_model1_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pre = _build_preprocessor(CATEGORICAL_FEATURES_PRE, NUMERIC_FEATURES_PRE)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
    }

    results = {}
    best_name, best_pipe, best_f1 = None, None, -1
    for name, clf in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        results[name] = metrics
        if metrics["f1"] > best_f1:
            best_f1, best_name, best_pipe = metrics["f1"], name, pipe

    _save_pipeline(best_pipe, out_path)
    return {"best_model": best_name, "all_results": results, "test_size": len(y_test)}


# ----------------------------------------------------------------------
# Modèle 2 : régression de la durée de retard
# ----------------------------------------------------------------------
def train_model2(df: pd.DataFrame, out_path: str = None):
    out_path = out_path or str(_ROOT / "models" / "model2_regression.joblib")
    X, y_raw = get_model2_data(df)
    # La distribution de la durée de retard est très asymétrique (longue
    # traîne jusqu'à >1700 min). On entraîne sur log1p(y) pour stabiliser
    # la variance, et on inverse (expm1) au moment de la prédiction.
    y = np.log1p(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    y_test_raw = np.expm1(y_test)
    pre = _build_preprocessor(CATEGORICAL_FEATURES_PRE, NUMERIC_FEATURES_PRE)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    results = {}
    best_name, best_pipe, best_mae = None, None, np.inf
    for name, reg in candidates.items():
        pipe = Pipeline([("pre", pre), ("reg", reg)])
        pipe.fit(X_train, y_train)
        y_pred_log = pipe.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        metrics = {
            "mae_minutes": mean_absolute_error(y_test_raw, y_pred),
            "rmse_minutes": np.sqrt(mean_squared_error(y_test_raw, y_pred)),
            "r2_log_scale": r2_score(y_test, y_pred_log),
        }
        results[name] = metrics
        if metrics["mae_minutes"] < best_mae:
            best_mae, best_name, best_pipe = metrics["mae_minutes"], name, pipe

    # on enveloppe le pipeline pour que la classe sache qu'elle doit
    # appliquer expm1 à la sortie (utilisé par l'app Streamlit)
    _save_pipeline(best_pipe, out_path, extra={"target_transform": "log1p"})
    return {"best_model": best_name, "all_results": results, "test_size": len(y_test)}


# ----------------------------------------------------------------------
# Modèle 3 : classification multi-classe du motif de retard
# ----------------------------------------------------------------------
def train_model3(df: pd.DataFrame, out_path: str = None):
    out_path = out_path or str(_ROOT / "models" / "model3_family.joblib")
    X, y = get_model3_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pre = _build_preprocessor(CATEGORICAL_FEATURES_M3, NUMERIC_FEATURES_M3)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RANDOM_STATE
        ),
    }

    results = {}
    best_name, best_pipe, best_f1 = None, None, -1
    label_classes = sorted(y.unique().tolist())

    for name, clf in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        if name == "XGBoost":
            # XGBoost classifier requires integer-encoded labels
            mapping = {c: i for i, c in enumerate(label_classes)}
            inv_mapping = {i: c for c, i in mapping.items()}
            y_train_enc = y_train.map(mapping)
            pipe.fit(X_train, y_train_enc)
            y_pred_enc = pipe.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(inv_mapping).values
        else:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }
        results[name] = metrics
        if metrics["f1_macro"] > best_f1:
            best_f1, best_name, best_pipe = metrics["f1_macro"], name, pipe
            best_label_mapping = mapping if name == "XGBoost" else None

    _save_pipeline(best_pipe, out_path, extra={
        "label_mapping": best_label_mapping, "classes": label_classes,
    })
    return {"best_model": best_name, "all_results": results, "test_size": len(y_test),
            "classes": label_classes}


def train_all(clean_data_path: str = None, report_path: str = None):
    clean_data_path = clean_data_path or str(_ROOT / "data" / "RAM_clean.parquet")
    report_path = report_path or str(_ROOT / "models" / "training_report.json")
    df = pd.read_parquet(clean_data_path)
    report = {
        "model1_binary": train_model1(df),
        "model2_regression": train_model2(df),
        "model3_family": train_model3(df),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    report = train_all()
    print(json.dumps(report, indent=2, default=str))
