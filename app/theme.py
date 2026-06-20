"""
Système de design centralisé — RAM Delay Intelligence.

Palette et typographie pensées pour un outil métier (BI compagnie
aérienne) : sobre, lisible, hiérarchie claire. Pas de gadget visuel —
la clarté des données prime.

Tokens
------
Couleur :
  --ram-bordeaux   #7A1530   couleur de marque, utilisée avec parcimonie
  --ram-bordeaux-d #5C0F24   variante foncée (hover, bordures)
  --ink             #1C1C21   texte principal
  --ink-soft        #5B5B63   texte secondaire / légendes
  --paper           #FFFFFF   fond des cartes
  --paper-tint      #F6F4F1   fond de page / sections alternées
  --line            #E3E0DA   séparateurs, bordures discrètes
  --gold            #C9A227   accent rare (mise en avant ponctuelle)
  --status-ok       #1E7A53   "à l'heure" / positif
  --status-warn     #C9A227   "retard modéré"
  --status-bad      #B23A2E   "retard sévère"

Typo : une seule famille sans-serif système (cohérence multiplateforme
Windows/Mac/Linux sans dépendance externe), avec une échelle de poids
disciplinée plutôt qu'un display face exotique — hors de propos pour un
outil de pilotage opérationnel.
"""

import streamlit as st

_FONT_STACK = (
    "'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', "
    "Arial, sans-serif"
)

GLOBAL_CSS = f"""
<style>
:root {{
    --ram-bordeaux: #7A1530;
    --ram-bordeaux-d: #5C0F24;
    --ink: #1C1C21;
    --ink-soft: #5B5B63;
    --paper: #FFFFFF;
    --paper-tint: #F6F4F1;
    --line: #E3E0DA;
    --gold: #C9A227;
    --status-ok: #1E7A53;
    --status-warn: #C9A227;
    --status-bad: #B23A2E;
}}

html, body, [class*="css"] {{
    font-family: {_FONT_STACK};
}}

/* ---- Fond général et conteneur principal ---- */
.stApp {{
    background-color: var(--paper-tint);
}}
.block-container {{
    padding-top: 1.6rem;
    max-width: 1180px;
}}

/* ---- Typographie générale ---- */
h1, h2, h3 {{
    color: var(--ink);
    font-weight: 600;
    letter-spacing: -0.01em;
}}
h1 {{ font-size: 1.85rem; margin-bottom: 0.15rem; }}
h2 {{ font-size: 1.3rem; margin-top: 1.6rem; }}
h3 {{ font-size: 1.05rem; }}
p, label, .stMarkdown {{ color: var(--ink); }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--ink-soft) !important; }}

/* ---- Bandeau d'en-tête de page (signature visuelle) ---- */
.ram-header {{
    background: linear-gradient(135deg, var(--ram-bordeaux) 0%, var(--ram-bordeaux-d) 100%);
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
.ram-header::after {{
    content: "";
    position: absolute;
    right: -40px; top: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
}}
.ram-header h1 {{
    color: #FFFFFF;
    margin: 0;
    font-size: 1.6rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}}
.ram-header .ram-subtitle {{
    color: rgba(255,255,255,0.82);
    font-size: 0.92rem;
    margin-top: 0.35rem;
    max-width: 720px;
    line-height: 1.45;
}}
.ram-header .ram-eyebrow {{
    color: var(--gold);
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    display: block;
}}

/* ---- Cartes KPI (st.metric) ---- */
div[data-testid="stMetric"] {{
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.95rem 1.1rem 0.85rem 1.1rem;
    box-shadow: 0 1px 2px rgba(28,28,33,0.04);
}}
div[data-testid="stMetric"] label {{
    color: var(--ink-soft) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
div[data-testid="stMetricValue"] {{
    color: var(--ink) !important;
    font-weight: 700 !important;
}}

/* ---- Onglets ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.4rem;
    border-bottom: 1px solid var(--line);
}}
.stTabs [data-baseweb="tab"] {{
    color: var(--ink-soft);
    font-weight: 600;
    padding: 0.6rem 0.9rem;
}}
.stTabs [aria-selected="true"] {{
    color: var(--ram-bordeaux) !important;
}}

/* ---- Boutons primaires ---- */
.stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"] {{
    background-color: var(--ram-bordeaux);
    border: none;
}}
.stButton button[kind="primary"]:hover, .stButton button[data-testid="baseButton-primary"]:hover {{
    background-color: var(--ram-bordeaux-d);
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background-color: var(--paper);
    border-right: 1px solid var(--line);
}}
section[data-testid="stSidebar"] h3 {{
    color: var(--ram-bordeaux);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 700;
}}

/* ---- Tableaux & dataframes ---- */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--line);
    border-radius: 8px;
}}

/* ---- Séparateurs ---- */
hr {{ border-color: var(--line); }}

/* ---- Expander ---- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {{
    font-weight: 600;
    color: var(--ink);
}}

/* ---- Badges de statut (utilisés via ram_badge()) ---- */
.ram-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
}}
.ram-badge.ok {{ background: rgba(30,122,83,0.1); color: var(--status-ok); }}
.ram-badge.warn {{ background: rgba(201,162,39,0.12); color: #8a6b14; }}
.ram-badge.bad {{ background: rgba(178,58,46,0.1); color: var(--status-bad); }}
</style>
"""

# Palette exposée pour usage direct dans les graphiques Plotly, afin que
# les couleurs des charts restent cohérentes avec le CSS ci-dessus.
COLORS = {
    "bordeaux": "#7A1530",
    "bordeaux_dark": "#5C0F24",
    "ink": "#1C1C21",
    "ink_soft": "#5B5B63",
    "paper": "#FFFFFF",
    "paper_tint": "#F6F4F1",
    "line": "#E3E0DA",
    "gold": "#C9A227",
    "ok": "#1E7A53",
    "warn": "#C9A227",
    "bad": "#B23A2E",
}

# Séquence de couleurs pour graphiques multi-catégories (cohérente, sans
# couleurs criardes — dérivée de la palette de marque + neutres).
CATEGORICAL_SEQUENCE = [
    "#7A1530", "#C9A227", "#3D6B8C", "#1E7A53", "#8A6B14",
    "#5B5B63", "#B23A2E", "#4A4A52", "#9C7A1E", "#2E4F66",
]


def inject_global_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str, eyebrow: str = ""):
    """Bandeau d'en-tête uniforme, signature visuelle répétée sur chaque page."""
    eyebrow_html = f'<span class="ram-eyebrow">{eyebrow}</span>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="ram-header">
            {eyebrow_html}
            <h1>{icon} {title}</h1>
            <div class="ram-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "ok") -> str:
    """Retourne le HTML d'un badge de statut (ok / warn / bad) à insérer
    dans un st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="ram-badge {kind}">{text}</span>'


def style_fig(fig):
    """Applique une mise en forme Plotly cohérente avec le thème de
    l'app : typographie, fond transparent, grille discrète. À appeler
    sur chaque figure juste avant st.plotly_chart()."""
    fig.update_layout(
        font_family="'Segoe UI', -apple-system, Arial, sans-serif",
        font_color=COLORS["ink"],
        title_font_size=15,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"])
    fig.update_yaxes(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"])
    return fig
