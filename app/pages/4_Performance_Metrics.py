import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from filters import render_filters, full_width_kwargs as fw
from theme import inject_global_css, page_header, style_fig, COLORS

st.set_page_config(page_title="Performance Metrics — RAM Delay Intelligence", page_icon="🎯", layout="wide")
inject_global_css()


@st.cache_data
def load_data():
    path = Path(__file__).parent.parent.parent / "data" / "RAM_clean.parquet"
    return pd.read_parquet(path)


page_header(
    icon="🎯", title="Performance Metrics",
    eyebrow="Module Performance Metrics",
    subtitle="Indicateurs clés de ponctualité : OTP strict, tolérance 15 "
             "minutes (OTP15), et OTP ajusté hors causes externes. "
             "Équivalent du module Performance Metrics du dashboard original.",
)

df_all = load_data()
df = render_filters(df_all, key_prefix="perf")

# Codes "hors contrôle de la compagnie" couramment exclus du KPI ajusté
# (équivalent ROTATION / AIRCRAFT CHANGE dans le rapport original)
EXCLUDED_FAMILIES_ADJUSTED = ["ROTATION", "ATC", "METEO"]

otp_strict = (df["retard_min"] == 0).mean() * 100
otp15 = (df["retard_min"] <= 15).mean() * 100
adjusted_mask = (df["retard_min"] <= 15) | (df["famille_retard"].isin(EXCLUDED_FAMILIES_ADJUSTED))
otp_adjusted = adjusted_mask.mean() * 100

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Percentage of On-Time Flights", f"{otp_strict:.2f}%")
with c2:
    st.metric("On-Time or Delay < 15 min (OTP15)", f"{otp15:.2f}%")
with c3:
    st.metric(
        "OTP ajusté (hors ATC/Météo/Rotation)", f"{otp_adjusted:.2f}%",
        help="Exclut les retards dus à des causes considérées hors contrôle "
             "direct de la compagnie (ATC, météo, rotation amont)."
    )

st.markdown("---")

c4, c5 = st.columns(2)
with c4:
    st.markdown("##### Percentage of On-Time Flights")
    fig1 = px.bar(x=["OTP strict"], y=[otp_strict], text=[f"{otp_strict:.1f}%"],
                  color_discrete_sequence=[COLORS["bordeaux"]])
    fig1.update_layout(yaxis_range=[0, 100], showlegend=False, xaxis_title="", yaxis_title="%")
    fig1.update_traces(textposition="outside", marker_line_width=0, width=0.4)
    st.plotly_chart(style_fig(fig1), **fw())
with c5:
    st.markdown("##### Percentage On-Time or Delay < 15 min")
    fig2 = px.bar(x=["OTP15"], y=[otp15], text=[f"{otp15:.1f}%"],
                  color_discrete_sequence=[COLORS["gold"]])
    fig2.update_layout(yaxis_range=[0, 100], showlegend=False, xaxis_title="", yaxis_title="%")
    fig2.update_traces(textposition="outside", marker_line_width=0, width=0.4)
    st.plotly_chart(style_fig(fig2), **fw())

st.markdown("---")
st.markdown("##### Évolution du taux de ponctualité dans le temps")
weekly_otp = (
    df.set_index("DAY_OF_ORIGIN")
    .resample("W")["retard_min"]
    .apply(lambda s: (s == 0).mean() * 100)
    .reset_index()
)
weekly_otp.columns = ["semaine", "otp_strict_pct"]
fig3 = px.line(weekly_otp, x="semaine", y="otp_strict_pct",
                labels={"semaine": "Semaine", "otp_strict_pct": "OTP strict (%)"},
                color_discrete_sequence=[COLORS["bordeaux"]], markers=True)
fig3.update_traces(line_width=2.5, marker_size=7)
st.plotly_chart(style_fig(fig3), **fw())

st.markdown("##### OTP par sous-type d'avion")
otp_by_subtype = (
    df.groupby("sous_type_avion")
    .agg(otp_strict=("retard_min", lambda s: (s == 0).mean() * 100),
         otp15=("retard_min", lambda s: (s <= 15).mean() * 100),
         nb_vols=("retard_min", "count"))
    .round(1)
    .sort_values("nb_vols", ascending=False)
    .reset_index()
)
st.dataframe(otp_by_subtype, **fw())
