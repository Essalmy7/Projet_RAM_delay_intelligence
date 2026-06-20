"""
Tests automatisés du projet RAM Delay Intelligence.

Exécution : pytest tests/test_app.py -v
(depuis la racine du projet)
"""

import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from streamlit.testing.v1 import AppTest

APP_DIR = ROOT / "app"
PAGES = [
    APP_DIR / "app.py",
    APP_DIR / "pages" / "1_Dashboard.py",
    APP_DIR / "pages" / "2_Analytics.py",
    APP_DIR / "pages" / "3_Weekly.py",
    APP_DIR / "pages" / "4_Performance_Metrics.py",
    APP_DIR / "pages" / "5_Predict.py",
]


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_page_runs_without_exception(page):
    at = AppTest.from_file(str(page), default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"{page.name} a levé une exception : {at.exception}"


def test_predict_pre_flight_button():
    at = AppTest.from_file(str(APP_DIR / "pages" / "5_Predict.py"), default_timeout=30)
    at.run()
    at.button[0].click().run()
    assert len(at.exception) == 0
    assert len(at.metric) >= 1


def test_predict_diagnose_button():
    at = AppTest.from_file(str(APP_DIR / "pages" / "5_Predict.py"), default_timeout=30)
    at.run()
    at.button[1].click().run()
    assert len(at.exception) == 0
    assert len(at.info) >= 1


def test_clean_dataset_has_no_nan_in_target_columns():
    df = pd.read_parquet(ROOT / "data" / "RAM_clean.parquet")
    for col in ["retard_min", "is_delayed", "famille_retard"]:
        assert df[col].isna().sum() == 0


def test_models_load_and_predict():
    df = pd.read_parquet(ROOT / "data" / "RAM_clean.parquet")
    sample = df.iloc[[0]]

    from model import load_pipeline
    from features import PRE_FLIGHT_FEATURES, POST_DELAY_EXTRA_FEATURES

    model1, _ = load_pipeline(ROOT / "models" / "model1_binary.joblib")
    pred1 = model1.predict(sample[PRE_FLIGHT_FEATURES])
    assert pred1[0] in (0, 1)

    model2, extra2 = load_pipeline(ROOT / "models" / "model2_regression.joblib")
    pred2 = model2.predict(sample[PRE_FLIGHT_FEATURES])
    assert pred2[0] is not None

    model3, extra3 = load_pipeline(ROOT / "models" / "model3_family.joblib")
    pred3 = model3.predict(sample[PRE_FLIGHT_FEATURES + POST_DELAY_EXTRA_FEATURES])
    assert pred3[0] is not None
    assert "classes" in extra3
    assert "label_mapping" in extra3


def test_xgboost_models_use_native_format_for_portability():
    """Les modèles dont le meilleur algorithme est XGBoost doivent être
    sauvegardés via le format natif XGBoost (.xgb.json), pas par pickle
    direct de l'objet Python — sinon le chargement casse en cas de
    changement de version majeure de xgboost ('flux d'entrée corrompu')."""
    for model_file in ["model1_binary.joblib", "model2_regression.joblib", "model3_family.joblib"]:
        payload = joblib.load(ROOT / "models" / model_file)
        assert "is_xgboost" in payload, f"{model_file}: format de sauvegarde inattendu"
        if payload["is_xgboost"]:
            xgb_path = ROOT / "models" / payload["xgb_model_path"]
            assert xgb_path.exists(), f"{model_file}: fichier booster natif manquant"
            assert xgb_path.suffix == ".json", f"{model_file}: le booster doit être en JSON"
    """Vérifie que la feature à haute cardinalité 'route' n'est pas
    utilisée par les modèles pré-vol (cf. README, section limites)."""
    from features import PRE_FLIGHT_FEATURES
    assert "route" not in PRE_FLIGHT_FEATURES


def test_no_route_feature_leakage():
    """Vérifie que la feature à haute cardinalité 'route' n'est pas
    utilisée par les modèles pré-vol (cf. README, section limites)."""
    from features import PRE_FLIGHT_FEATURES
    assert "route" not in PRE_FLIGHT_FEATURES


def test_theme_css_injected_on_all_pages():
    """Chaque page doit injecter le CSS global et afficher le bandeau
    d'en-tête stylisé (signature visuelle de l'app)."""
    for page in PAGES:
        at = AppTest.from_file(str(page), default_timeout=30)
        at.run()
        mds = [m.value for m in at.markdown]
        has_css = any("ram-header" in m for m in mds)
        assert has_css, f"{page.name}: bandeau d'en-tête stylisé manquant"


def test_missing_xgboost_shows_friendly_error():
    """Si xgboost n'est pas installé, l'app doit afficher un message
    d'erreur clair (st.error + st.stop) plutôt qu'un traceback brut.
    Exécuté dans un sous-processus isolé : une fois xgboost importé par
    un autre test dans ce process, joblib garde une référence interne
    qui rendrait un blocage via sys.meta_path inefficace ici."""
    script = f"""
import sys
sys.path.append({str(ROOT / "src")!r})

class BlockXGBoost:
    def find_spec(self, name, path, target=None):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ModuleNotFoundError(
                f"No module named '{{name.split('.')[0]}}'", name=name.split(".")[0]
            )
        return None

sys.meta_path.insert(0, BlockXGBoost())

from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(APP_DIR / "pages" / "5_Predict.py")!r}, default_timeout=30)
at.run()
assert len(at.exception) == 0, at.exception
assert len(at.error) >= 1, "Aucun message d'erreur affiché"
assert "xgboost" in at.error[0].value.lower()
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout
