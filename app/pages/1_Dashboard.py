import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from filters import render_filters, export_button, full_width_kwargs as fw
from theme import inject_global_css, page_header, style_fig, COLORS, CATEGORICAL_SEQUENCE

st.set_page_config(page_title="Dashboard — RAM Delay Intelligence", page_icon="📊", layout="wide")
inject_global_css()


@st.cache_data
def load_data():
    path = Path(__file__).parent.parent.parent / "data" / "RAM_clean.parquet"
    return pd.read_parquet(path)


page_header(
    icon="📊", title="Dashboard",
    eyebrow="Module Overview",
    subtitle="Vision globale des vols retardés : répartition par sous-type d'avion, "
             "sévérité du retard et distribution temporelle. Équivalent du module "
             "Overview du dashboard original.",
)

df_all = load_data()
df = render_filters(df_all, key_prefix="dash")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Vols", f"{len(df):,}".replace(",", " "))
with col2:
    st.metric("Vols retardés", f"{df['is_delayed'].sum():,}".replace(",", " "))
with col3:
    st.metric("Taux de retard", f"{df['is_delayed'].mean()*100:.1f}%")
with col4:
    st.metric("Retard moyen (si retard)", f"{df.loc[df['is_delayed']==1,'retard_min'].mean():.0f} min")

st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.markdown("##### Répartition des retards par sous-type d'avion")
    sub = (
        df[df["is_delayed"] == 1]
        .groupby("sous_type_avion")
        .size()
        .reset_index(name="nb_vols_retardes")
        .sort_values("nb_vols_retardes", ascending=False)
    )
    sub["pct"] = sub["nb_vols_retardes"] / sub["nb_vols_retardes"].sum() * 100
    fig = px.bar(
        sub, x="sous_type_avion", y="pct", text=sub["pct"].round(1).astype(str) + "%",
        labels={"sous_type_avion": "Sous-type avion", "pct": "% des vols retardés"},
        color_discrete_sequence=[COLORS["bordeaux"]],
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    st.plotly_chart(style_fig(fig), **fw())

with c2:
    st.markdown("##### Sévérité des retards (≤ ou > 15 min)")
    delayed = df[df["is_delayed"] == 1].copy()
    delayed["severite"] = delayed["retard_min"].apply(lambda r: "≤ 15 min" if r <= 15 else "> 15 min")
    sev = delayed["severite"].value_counts().reset_index()
    sev.columns = ["severite", "count"]
    fig2 = px.pie(
        sev, names="severite", values="count", hole=0.55,
        color_discrete_sequence=[COLORS["gold"], COLORS["bordeaux"]],
    )
    fig2.update_traces(marker_line_color="white", marker_line_width=2)
    st.plotly_chart(style_fig(fig2), **fw())

st.markdown("##### Distribution des vols retardés par tranche de retard")
tranche_order = ["À l'heure", "0-15 min", "16-60 min", "1h-2h", "2h-4h", "> 4h"]
tranche_counts = df["tranche_retard"].value_counts().reindex(tranche_order).fillna(0).reset_index()
tranche_counts.columns = ["tranche", "count"]
fig3 = px.bar(
    tranche_counts, x="tranche", y="count",
    labels={"tranche": "Tranche de retard", "count": "Nombre de vols"},
    color_discrete_sequence=[COLORS["bordeaux"]],
)
fig3.update_traces(marker_line_width=0)
st.plotly_chart(style_fig(fig3), **fw())

st.markdown("##### Évolution du taux de retard (par jour)")
daily = (
    df.groupby(df["DAY_OF_ORIGIN"].dt.date)["is_delayed"]
    .agg(["sum", "count"])
    .reset_index()
)
daily.columns = ["date", "vols_retardes", "total_vols"]
daily["taux_retard"] = daily["vols_retardes"] / daily["total_vols"] * 100
fig4 = px.line(
    daily, x="date", y="taux_retard",
    labels={"date": "Date", "taux_retard": "Taux de retard (%)"},
    color_discrete_sequence=[COLORS["bordeaux"]],
)
fig4.update_traces(line_width=2.5)
st.plotly_chart(style_fig(fig4), **fw())

st.markdown("---")
st.markdown("### Données détaillées")
display_cols = [
    "DAY_OF_ORIGIN", "FN_NUMBER", "sous_type_avion", "immatriculation",
    "aeroport_dep", "aeroport_arr", "retard_min", "famille_retard",
]
st.dataframe(df[display_cols].sort_values("DAY_OF_ORIGIN", ascending=False), **fw(), height=300)
export_button(df[display_cols], "dashboard_export.csv", key="dash_export")
