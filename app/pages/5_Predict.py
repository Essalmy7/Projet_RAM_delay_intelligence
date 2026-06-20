import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from filters import full_width_kwargs as fw
from theme import inject_global_css, page_header, style_fig, COLORS

st.set_page_config(page_title="Predict — RAM Delay Intelligence", page_icon="🤖", layout="wide")
inject_global_css()

ROOT = Path(__file__).parent.parent.parent


@st.cache_data
def load_data():
    return pd.read_parquet(ROOT / "data" / "RAM_clean.parquet")


@st.cache_resource
def load_models():
    try:
        from model import load_pipeline

        m1, _ = load_pipeline(ROOT / "models" / "model1_binary.joblib")
        m2_pipe, m2_extra = load_pipeline(ROOT / "models" / "model2_regression.joblib")
        m3_pipe, m3_extra = load_pipeline(ROOT / "models" / "model3_family.joblib")
        m2_bundle = {"pipeline": m2_pipe, **m2_extra}
        m3_bundle = {"pipeline": m3_pipe, **m3_extra}
        return m1, m2_bundle, m3_bundle
    except ModuleNotFoundError as e:
        st.error(
            f"❌ Dépendance manquante pour charger les modèles : **{e.name}**.\n\n"
            "Les modèles ont été entraînés avec `xgboost` (et `scikit-learn`). "
            "Installez les dépendances du projet avec :\n\n"
            "```bash\npip install -r requirements.txt\n```\n\n"
            "Si le problème persiste, vérifiez que vous utilisez le même "
            "environnement Python (`pip` / `python`) que celui dans lequel "
            "vous avez exécuté cette commande."
        )
        st.stop()
    except Exception as e:
        st.error(
            f"❌ Erreur lors du chargement des modèles : **{type(e).__name__}** — {e}\n\n"
            "Si l'erreur mentionne XGBoost ou un format corrompu, essayez de "
            "réentraîner les modèles localement avec votre version installée :\n\n"
            "```bash\npython src/data_processing.py\npython src/model.py\n```"
        )
        st.stop()


df = load_data()
model1, model2_bundle, model3_bundle = load_models()
model2 = model2_bundle["pipeline"]
model3 = model3_bundle["pipeline"]
model3_classes = model3_bundle["classes"]
model3_mapping = model3_bundle["label_mapping"]

page_header(
    icon="🤖", title="Predict",
    eyebrow="Module Machine Learning — ajouté au dashboard original",
    subtitle="Deux temporalités : avant le vol (retard probable, durée "
             "estimée) et après constat du retard (diagnostic du motif le "
             "plus probable, en aide à la saisie pour l'opérateur).",
)

tab1, tab2 = st.tabs(["✈️  Avant le vol", "🔍  Diagnostic du motif"])

# ----------------------------------------------------------------------
# Onglet 1 : prédiction pré-vol (Modèle 1 + Modèle 2)
# ----------------------------------------------------------------------
with tab1:
    st.markdown("#### Simuler un vol")
    st.caption(
        "Renseignez les caractéristiques connues avant le décollage. Le "
        "modèle estime la probabilité de retard, puis (si retard probable) "
        "sa durée."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sous_type = st.selectbox("Sous-type d'avion", sorted(df["sous_type_avion"].unique()))
        compagnie = st.selectbox("Compagnie opératrice", sorted(df["compagnie_operatrice"].unique()))
        type_courrier = st.selectbox("Type de courrier", [v for v in sorted(df["type_courrier"].unique()) if v != "-"])
    with c2:
        secteur_origine = st.selectbox("Secteur d'origine", [v for v in sorted(df["secteur_origine"].unique()) if v != "-"])
        secteur_dest = st.selectbox("Secteur de destination", [v for v in sorted(df["secteur_destination"].unique()) if v != "-"])
        trafic = st.selectbox("Trafic", [v for v in sorted(df["trafic_interne_externe"].unique()) if v != "Non renseigné"])
    with c3:
        jour = st.selectbox("Jour de la semaine", ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"])
        mois = st.selectbox("Mois", sorted(df["mois"].unique()))
        periode = st.selectbox("Période de la journée", ["Nuit (00h-06h)", "Matin (06h-12h)", "Après-midi (12h-18h)", "Soir (18h-24h)"])

    c4, c5 = st.columns(2)
    with c4:
        heure_dep = st.slider("Heure de départ programmée", 0, 23, 8)
    with c5:
        duree_vol = st.slider("Temps de vol programmé (minutes)", 25, 860, 120)

    is_weekend = jour in ("samedi", "dimanche")

    input_row = pd.DataFrame([{
        "sous_type_avion": sous_type,
        "compagnie_operatrice": compagnie,
        "type_courrier": type_courrier,
        "secteur_origine": secteur_origine,
        "secteur_destination": secteur_dest,
        "jour_semaine": jour,
        "is_weekend": is_weekend,
        "mois": mois,
        "periode_journee": periode,
        "dep_hour_sched": heure_dep,
        "temps_vol_sched_min": duree_vol,
        "trafic_interne_externe": trafic,
    }])

    st.markdown("")
    predict_clicked = st.button("🔮  Prédire", type="primary", key="predict_pre_flight", use_container_width=True)

    if predict_clicked:
        proba_delay = model1.predict_proba(input_row)[0, 1]
        pred_delay = model1.predict(input_row)[0]

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilité de retard", f"{proba_delay*100:.1f}%")
            if pred_delay == 1:
                st.error("⚠️  Vol probablement **EN RETARD**")
            else:
                st.success("✅  Vol probablement **À L'HEURE**")
        with c2:
            fig = px.pie(
                values=[proba_delay, 1 - proba_delay],
                names=["Retard", "À l'heure"],
                hole=0.6,
                color_discrete_sequence=[COLORS["bordeaux"], COLORS["ok"]],
            )
            fig.update_traces(marker_line_color="white", marker_line_width=2)
            st.plotly_chart(style_fig(fig), **fw())

        if pred_delay == 1:
            duree_pred_log = model2.predict(input_row)[0]
            duree_pred = np.expm1(duree_pred_log)
            st.warning(f"⏱️  Durée de retard estimée : **{duree_pred:.0f} minutes**")
            st.caption(
                "Note méthodologique : la durée de retard dépend fortement de "
                "causes imprévisibles au moment de la prédiction (panne précise, "
                "décision ATC...). Cette estimation a une marge d'erreur "
                "significative (MAE ≈ 21 min) ; elle indique un ordre de "
                "grandeur, pas une valeur certaine."
            )

# ----------------------------------------------------------------------
# Onglet 2 : diagnostic du motif (Modèle 3), une fois le retard constaté
# ----------------------------------------------------------------------
with tab2:
    st.markdown("#### Diagnostiquer le motif d'un retard déjà constaté")
    st.caption(
        "Ce modèle s'utilise après qu'un retard a été observé : il suggère "
        "le motif (famille) le plus probable, comme aide à la saisie pour "
        "l'opérateur — qui valide ou corrige. Il répond directement au "
        "problème de fiabilité de saisie identifié dans le rapport de stage "
        "original."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sous_type2 = st.selectbox("Sous-type d'avion", sorted(df["sous_type_avion"].unique()), key="m3_subtype")
        compagnie2 = st.selectbox("Compagnie opératrice", sorted(df["compagnie_operatrice"].unique()), key="m3_compagnie")
        type_courrier2 = st.selectbox("Type de courrier", [v for v in sorted(df["type_courrier"].unique()) if v != "-"], key="m3_courrier")
    with c2:
        secteur_origine2 = st.selectbox("Secteur d'origine", [v for v in sorted(df["secteur_origine"].unique()) if v != "-"], key="m3_sec_o")
        secteur_dest2 = st.selectbox("Secteur de destination", [v for v in sorted(df["secteur_destination"].unique()) if v != "-"], key="m3_sec_d")
        trafic2 = st.selectbox("Trafic", [v for v in sorted(df["trafic_interne_externe"].unique()) if v != "Non renseigné"], key="m3_trafic")
    with c3:
        jour2 = st.selectbox("Jour de la semaine", ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"], key="m3_jour")
        mois2 = st.selectbox("Mois", sorted(df["mois"].unique()), key="m3_mois")
        periode2 = st.selectbox("Période de la journée", ["Nuit (00h-06h)", "Matin (06h-12h)", "Après-midi (12h-18h)", "Soir (18h-24h)"], key="m3_periode")

    c4, c5, c6 = st.columns(3)
    with c4:
        heure_dep2 = st.slider("Heure de départ programmée", 0, 23, 8, key="m3_heure")
    with c5:
        duree_vol2 = st.slider("Temps de vol programmé (min)", 25, 860, 120, key="m3_duree_vol")
    with c6:
        retard_constate = st.slider("Retard constaté (min)", 1, 500, 30, key="m3_retard")

    def tranche(r):
        if r <= 0: return "À l'heure"
        if r <= 15: return "0-15 min"
        if r <= 60: return "16-60 min"
        if r <= 120: return "1h-2h"
        if r <= 240: return "2h-4h"
        return "> 4h"

    input_row3 = pd.DataFrame([{
        "sous_type_avion": sous_type2,
        "compagnie_operatrice": compagnie2,
        "type_courrier": type_courrier2,
        "secteur_origine": secteur_origine2,
        "secteur_destination": secteur_dest2,
        "jour_semaine": jour2,
        "is_weekend": jour2 in ("samedi", "dimanche"),
        "mois": mois2,
        "periode_journee": periode2,
        "dep_hour_sched": heure_dep2,
        "temps_vol_sched_min": duree_vol2,
        "trafic_interne_externe": trafic2,
        "retard_min": retard_constate,
        "tranche_retard": tranche(retard_constate),
    }])

    st.markdown("")
    diagnose_clicked = st.button("🔍  Diagnostiquer le motif", type="primary", key="predict_family", use_container_width=True)

    if diagnose_clicked:
        proba = model3.predict_proba(input_row3)[0]
        inv_mapping = {v: k for k, v in model3_mapping.items()}
        classes_ordered = [inv_mapping[i] for i in range(len(inv_mapping))]
        result = pd.DataFrame({"famille": classes_ordered, "probabilite": proba}).sort_values(
            "probabilite", ascending=False
        )

        st.markdown("---")
        top = result.iloc[0]
        st.info(f"**Motif le plus probable : {top['famille']}** ({top['probabilite']*100:.1f}% de confiance)")

        fig = px.bar(
            result.head(6), x="probabilite", y="famille", orientation="h",
            labels={"probabilite": "Probabilité", "famille": "Famille de motif"},
            color_discrete_sequence=[COLORS["bordeaux"]],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig), **fw())
        st.caption(
            "Suggestion à valider par l'opérateur — accuracy globale du modèle "
            "≈ 56% sur 9 familles, ce qui reste nettement supérieur au hasard "
            "(≈ 11% pour 9 classes) mais ne remplace pas le jugement humain."
        )

st.markdown("---")
with st.expander("📈  Performance des modèles (jeu de test)"):
    with open(ROOT / "models" / "training_report.json") as f:
        report = json.load(f)

    st.markdown("**Modèle 1 — Classification binaire (à l'heure / en retard)**")
    df1 = pd.DataFrame(report["model1_binary"]["all_results"]).T.round(3)
    st.dataframe(df1, **fw())

    st.markdown("**Modèle 2 — Régression de la durée de retard**")
    df2 = pd.DataFrame(report["model2_regression"]["all_results"]).T.round(3)
    st.dataframe(df2, **fw())

    st.markdown("**Modèle 3 — Classification du motif (famille)**")
    df3 = pd.DataFrame(report["model3_family"]["all_results"]).T.round(3)
    st.dataframe(df3, **fw())
