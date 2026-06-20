import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from filters import render_filters, export_button, full_width_kwargs as fw
from theme import inject_global_css, page_header, style_fig, COLORS, CATEGORICAL_SEQUENCE

st.set_page_config(page_title="Analytics — RAM Delay Intelligence", page_icon="🔬", layout="wide")
inject_global_css()


@st.cache_data
def load_data():
    path = Path(__file__).parent.parent.parent / "data" / "RAM_clean.parquet"
    return pd.read_parquet(path)


page_header(
    icon="🔬", title="Analytics",
    eyebrow="Module Analytics",
    subtitle="Analyses approfondies des causes de retard, par famille de motif "
             "(ATC, ROTATION, TECHNIQUE, AVARIE...). Équivalent du module "
             "Analytics du dashboard original.",
)

df_all = load_data()
df = render_filters(df_all, key_prefix="analytics")

delayed = df[df["is_delayed"] == 1].copy()
# On exclut "AUCUN" (motif non renseigné) des analyses de causes : ce
# n'est pas une vraie famille, c'est un manque de saisie — point déjà
# identifié dans le rapport de stage original.
delayed_documented = delayed[delayed["famille_retard"] != "AUCUN"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vols retardés (total)", f"{len(delayed):,}".replace(",", " "))
with col2:
    st.metric("Avec motif renseigné", f"{len(delayed_documented):,}".replace(",", " "))
with col3:
    pct_undocumented = (1 - len(delayed_documented) / max(len(delayed), 1)) * 100
    st.metric("Motif NON renseigné", f"{pct_undocumented:.1f}%",
              help="Vols en retard sans code motif dans NETLINE/AMOS — "
                   "problème de fiabilité de saisie identifié dans le rapport original.")

st.markdown("---")

st.markdown("##### Répartition des retards par famille de codes")
fam = delayed_documented["famille_retard"].value_counts().reset_index()
fam.columns = ["famille", "count"]
fam["pct"] = fam["count"] / fam["count"].sum() * 100
fig = px.bar(
    fam, x="famille", y="pct", text=fam["pct"].round(1).astype(str) + "%",
    labels={"famille": "Famille de motif", "pct": "% des retards documentés"},
    color_discrete_sequence=[COLORS["bordeaux"]],
)
fig.update_traces(textposition="outside", marker_line_width=0)
st.plotly_chart(style_fig(fig), **fw())

st.markdown("---")
st.markdown("### Détail par famille")

fam_options = sorted(delayed_documented["famille_retard"].unique())
sel_fam = st.selectbox("Choisir une famille à détailler", fam_options)
sub = delayed_documented[delayed_documented["famille_retard"] == sel_fam]

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Durée moyenne de retard — {sel_fam}**")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Retard moyen", f"{sub['retard_min'].mean():.0f} min")
    with m2:
        st.metric("Retard médian", f"{sub['retard_min'].median():.0f} min")
    with m3:
        st.metric("Nb vols", f"{len(sub):,}".replace(",", " "))
with c2:
    by_subtype = sub["sous_type_avion"].value_counts().reset_index()
    by_subtype.columns = ["sous_type_avion", "count"]
    fig_sub = px.pie(
        by_subtype, names="sous_type_avion", values="count",
        title=f"Répartition par sous-type — {sel_fam}",
        color_discrete_sequence=CATEGORICAL_SEQUENCE,
        hole=0.45,
    )
    fig_sub.update_traces(marker_line_color="white", marker_line_width=1.5)
    st.plotly_chart(style_fig(fig_sub), **fw())

st.markdown(f"##### Distribution horaire des retards — {sel_fam}")
fig_hour = px.histogram(
    sub, x="periode_journee", category_orders={
        "periode_journee": ["Nuit (00h-06h)", "Matin (06h-12h)", "Après-midi (12h-18h)", "Soir (18h-24h)"]
    },
    labels={"periode_journee": "Période de la journée"},
    color_discrete_sequence=[COLORS["gold"]],
)
fig_hour.update_traces(marker_line_width=0)
st.plotly_chart(style_fig(fig_hour), **fw())

st.markdown("---")
st.markdown("### Tableau récapitulatif par famille")
summary = (
    delayed_documented.groupby("famille_retard")
    .agg(nb_vols=("retard_min", "count"), retard_moyen=("retard_min", "mean"), retard_median=("retard_min", "median"))
    .round(1)
    .sort_values("nb_vols", ascending=False)
    .reset_index()
)
st.dataframe(summary, **fw())
export_button(summary, "analytics_export.csv", key="analytics_export")
