"""Composants de filtrage partagés par les pages Dashboard / Analytics / Weekly / Performance."""

import streamlit as st
import pandas as pd
from packaging.version import Version

# Compatibilité multi-versions Streamlit : `use_container_width` est
# accepté jusqu'à 1.x mais déprécié au profit de `width="stretch"` dans
# les versions plus récentes (le paramètre legacy peut être retiré selon
# la version installée chez l'utilisateur). On détecte dynamiquement la
# bonne API plutôt que de figer un choix qui casse sur certaines versions.
_USE_WIDTH_STRING = Version(st.__version__) >= Version("1.50.0")


def full_width_kwargs() -> dict:
    """Retourne le bon kwarg ('width' ou 'use_container_width') selon la
    version de Streamlit installée, pour st.dataframe / st.plotly_chart."""
    if _USE_WIDTH_STRING:
        return {"width": "stretch"}
    return {"use_container_width": True}


def render_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Affiche un bloc de filtres dans la sidebar et retourne le dataframe filtré."""
    st.sidebar.markdown("### 🔎 Filtres")

    subtypes = sorted(df["sous_type_avion"].unique())
    sel_subtypes = st.sidebar.multiselect(
        "Sous-type d'avion", subtypes, default=[], key=f"{key_prefix}_subtype"
    )

    matricules = sorted(df["immatriculation"].unique())
    sel_matricules = st.sidebar.multiselect(
        "Immatriculation", matricules, default=[], key=f"{key_prefix}_matricule"
    )

    families = sorted(df["famille_retard"].unique())
    sel_families = st.sidebar.multiselect(
        "Famille de motif de retard", families, default=[], key=f"{key_prefix}_family"
    )

    min_date = df["DAY_OF_ORIGIN"].min().date()
    max_date = df["DAY_OF_ORIGIN"].max().date()
    date_range = st.sidebar.date_input(
        "Période", value=(min_date, max_date), min_value=min_date, max_value=max_date,
        key=f"{key_prefix}_date"
    )

    filtered = df.copy()
    if sel_subtypes:
        filtered = filtered[filtered["sous_type_avion"].isin(sel_subtypes)]
    if sel_matricules:
        filtered = filtered[filtered["immatriculation"].isin(sel_matricules)]
    if sel_families:
        filtered = filtered[filtered["famille_retard"].isin(sel_families)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["DAY_OF_ORIGIN"].dt.date >= start)
            & (filtered["DAY_OF_ORIGIN"].dt.date <= end)
        ]

    n = len(filtered)
    pct = n / max(len(df), 1) * 100
    st.sidebar.markdown(
        f"""<div style="background:#F6F4F1; border:1px solid #E3E0DA; border-radius:8px;
        padding:0.55rem 0.7rem; margin-top:0.4rem; font-size:0.85rem;">
        <b>{n:,}</b> vol(s) sélectionné(s) <span style="color:#5B5B63;">({pct:.0f}% du total)</span>
        </div>""".replace(",", " "),
        unsafe_allow_html=True,
    )
    return filtered


def export_button(df: pd.DataFrame, filename: str, key: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter en CSV", data=csv, file_name=filename, mime="text/csv", key=key
    )
