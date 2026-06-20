import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from filters import render_filters, export_button, full_width_kwargs as fw
from theme import inject_global_css, page_header, style_fig, COLORS

st.set_page_config(page_title="Weekly — RAM Delay Intelligence", page_icon="📅", layout="wide")
inject_global_css()

DAY_ORDER = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


@st.cache_data
def load_data():
    path = Path(__file__).parent.parent.parent / "data" / "RAM_clean.parquet"
    return pd.read_parquet(path)


page_header(
    icon="📅", title="Weekly",
    eyebrow="Module Weekly",
    subtitle="Tableau croisé dynamique des codes de retard par jour de la "
             "semaine, pour identifier les jours les plus problématiques. "
             "Équivalent du module Weekly du dashboard original.",
)

df_all = load_data()
df = render_filters(df_all, key_prefix="weekly")
delayed = df[(df["is_delayed"] == 1) & (df["famille_retard"] != "AUCUN")].copy()

st.markdown("### Tableau croisé : famille de motif × jour de la semaine")
pivot = pd.crosstab(delayed["famille_retard"], delayed["jour_semaine"])
pivot = pivot.reindex(columns=DAY_ORDER, fill_value=0)
pivot["TOTAL"] = pivot.sum(axis=1)
pivot = pivot.sort_values("TOTAL", ascending=False)
st.dataframe(pivot, **fw())
export_button(pivot.reset_index(), "weekly_export.csv", key="weekly_export")

st.markdown("---")

st.markdown("##### Taux de retard par jour de la semaine")
by_day = (
    df.groupby("jour_semaine", observed=True)["is_delayed"]
    .agg(["mean", "count"])
    .reindex(DAY_ORDER)
    .reset_index()
)
by_day.columns = ["jour", "taux_retard", "nb_vols"]
by_day["taux_retard_pct"] = by_day["taux_retard"] * 100
fig = px.bar(
    by_day, x="jour", y="taux_retard_pct",
    text=by_day["taux_retard_pct"].round(1).astype(str) + "%",
    labels={"jour": "Jour de la semaine", "taux_retard_pct": "Taux de retard (%)"},
    color_discrete_sequence=[COLORS["bordeaux"]],
)
fig.update_traces(textposition="outside", marker_line_width=0)
st.plotly_chart(style_fig(fig), **fw())

st.markdown("##### Jour le plus problématique par famille de motif")
top_day_per_family = (
    delayed.groupby(["famille_retard", "jour_semaine"], observed=True)
    .size()
    .reset_index(name="count")
    .sort_values(["famille_retard", "count"], ascending=[True, False])
    .groupby("famille_retard")
    .first()
    .reset_index()
    .sort_values("count", ascending=False)
)
top_day_per_family.columns = ["Famille", "Jour le plus fréquent", "Nb occurrences"]
st.dataframe(top_day_per_family, **fw())

st.markdown("##### Heatmap famille × jour")
heat = pivot.drop(columns="TOTAL")
fig2 = px.imshow(
    heat, labels=dict(x="Jour de la semaine", y="Famille de motif", color="Nb vols"),
    color_continuous_scale=[[0, COLORS["paper_tint"]], [1, COLORS["bordeaux"]]],
    aspect="auto",
)
st.plotly_chart(style_fig(fig2), **fw())
