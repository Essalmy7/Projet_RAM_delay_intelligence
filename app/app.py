"""
RAM Delay Intelligence — Application Streamlit
Point d'entrée principal. Configure la page et affiche l'accueil.
Les 5 modules (Dashboard, Analytics, Weekly, Performance, Predict) sont
des pages séparées dans app/pages/, conformément à la structure
multi-pages native de Streamlit.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))
from theme import inject_global_css, page_header, COLORS

st.set_page_config(
    page_title="RAM Delay Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


@st.cache_data
def load_data():
    path = Path(__file__).parent.parent / "data" / "RAM_clean.parquet"
    return pd.read_parquet(path)


MODULES = [
    ("📊", "Dashboard", "Vue d'ensemble : répartition des retards par sous-type, sévérité, distribution temporelle.", "1_Dashboard"),
    ("🔬", "Analytics", "Analyse des causes par famille de motif (ATC, ROTATION, TECHNIQUE...).", "2_Analytics"),
    ("📅", "Weekly", "Tableau croisé motif × jour de semaine, heatmap des tendances hebdomadaires.", "3_Weekly"),
    ("🎯", "Performance Metrics", "KPIs de ponctualité : OTP strict, OTP15, OTP ajusté.", "4_Performance_Metrics"),
    ("🤖", "Predict (ML)", "Modèles prédictifs : retard probable, durée estimée, diagnostic du motif.", "5_Predict"),
]


def main():
    df = load_data()

    page_header(
        icon="✈️",
        title="RAM Delay Intelligence",
        eyebrow="Royal Air Maroc — Pôle Exploitation",
        subtitle=(
            "Plateforme d'analyse et de prédiction de la ponctualité des flottes. "
            "Centralise les données NETLINE / AMOS, fiabilise le suivi des motifs "
            "de retard, et anticipe les retards à venir."
        ),
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vols analysés", f"{len(df):,}".replace(",", " "))
    with col2:
        pct_on_time = (1 - df["is_delayed"].mean()) * 100
        st.metric("Taux de ponctualité", f"{pct_on_time:.1f}%")
    with col3:
        avg_delay = df.loc[df["is_delayed"] == 1, "retard_min"].mean()
        st.metric("Retard moyen (vols retardés)", f"{avg_delay:.0f} min")
    with col4:
        period = f"{df['DAY_OF_ORIGIN'].min():%d/%m/%y} → {df['DAY_OF_ORIGIN'].max():%d/%m/%y}"
        st.metric("Période couverte", period)

    st.markdown("##  ")
    st.markdown("### Modules disponibles")
    st.caption("Utilisez le menu à gauche pour naviguer, ou ouvrez directement un module ci-dessous.")

    cols = st.columns(3)
    for i, (icon, name, desc, page_file) in enumerate(MODULES):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### {icon} {name}")
                st.caption(desc)
                st.page_link(f"pages/{page_file}.py", label="Ouvrir →", use_container_width=True)

    st.markdown("---")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### À propos de ce projet")
        st.markdown(
            """
Dans le cadre d’un projet académique en collaboration avec Royal Air Maroc, le travail s’est concentré sur l’analyse de la ponctualité des flottes aériennes et sur l’exploitation des données opérationnelles liées aux vols et aux retards.

La ponctualité constitue un indicateur majeur de performance dans le transport aérien. Les retards peuvent être liés à plusieurs facteurs : rotation des avions, contraintes de maintenance, contrôle aérien, correspondances, avaries, traitement avion ou encore contraintes de programmation.

La difficulté principale réside dans la consolidation et l’exploitation de données opérationnelles issues de plusieurs systèmes, notamment les données de suivi des vols et les données liées à la maintenance.

La problématique traitée est la suivante : Comment centraliser, fiabiliser et exploiter les données opérationnelles afin d'améliorer le suivi de la ponctualité des vols, analyser les causes de retard et anticiper les retards à venir ?
            """
        )

    with right:
        st.markdown("### Note méthodologique")
        st.markdown(
            f"""
<div style="background:{COLORS['paper']}; border:1px solid {COLORS['line']};
border-radius:10px; padding:1rem 1.2rem; font-size:0.92rem; line-height:1.55;">
Le jeu de données contient <b>17&nbsp;017 segments de vol</b> opérés entre le
29&nbsp;mars et le 24&nbsp;juin 2025, consolidés à partir des systèmes
<b>NETLINE</b> (vols temps réel) et <b>AMOS</b> (maintenance technique).
</div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
