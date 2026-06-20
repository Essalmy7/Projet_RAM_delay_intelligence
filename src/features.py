"""
Préparation des jeux de features pour les 3 modèles:
  - Modèle 1: classification binaire (vol à l'heure / en retard)
  - Modèle 2: régression (durée du retard, sur les vols en retard)
  - Modèle 3: classification multi-classe (famille de motif, sur les vols en retard)

IMPORTANT - fuite de données (data leakage):
Les modèles 1 et 2 sont des modèles "pré-vol": ils ne doivent utiliser QUE
des variables connues avant le décollage (avion, route, horaire programmé,
jour, historique). Toute variable mesurée pendant/après le vol (temps de
vol réalisé, taxi time réel, motif de retard...) est exclue.
Le modèle 3 intervient une fois le retard déjà constaté: il peut donc
utiliser en plus la durée du retard et le moment où il est observé.
"""

import pandas as pd
import numpy as np

# Features disponibles avant le décollage (modèles 1 et 2)
PRE_FLIGHT_FEATURES = [
    "sous_type_avion",
    "compagnie_operatrice",
    "type_courrier",
    "secteur_origine",
    "secteur_destination",
    "jour_semaine",
    "is_weekend",
    "mois",
    "periode_journee",
    "dep_hour_sched",
    "temps_vol_sched_min",
    "trafic_interne_externe",
]

# Features additionnelles disponibles une fois le retard constaté (modèle 3)
POST_DELAY_EXTRA_FEATURES = [
    "retard_min",
    "tranche_retard",
]

TARGET_BINARY = "is_delayed"
TARGET_REGRESSION = "retard_min"
TARGET_FAMILY = "famille_retard"

CATEGORICAL_FEATURES_PRE = [
    "sous_type_avion", "compagnie_operatrice", "type_courrier",
    "secteur_origine", "secteur_destination",
    "jour_semaine", "mois", "periode_journee", "trafic_interne_externe",
]
NUMERIC_FEATURES_PRE = ["is_weekend", "dep_hour_sched", "temps_vol_sched_min"]


def get_model1_data(df: pd.DataFrame):
    """Classification binaire: à l'heure (0) vs en retard (1)."""
    X = df[PRE_FLIGHT_FEATURES].copy()
    y = df[TARGET_BINARY].copy()
    return X, y


def get_model2_data(df: pd.DataFrame):
    """Régression de la durée de retard, restreinte aux vols en retard
    (is_delayed == 1), pour ne pas biaiser le modèle avec la masse de
    zéros qui appartient déjà au modèle 1."""
    sub = df[df[TARGET_BINARY] == 1]
    X = sub[PRE_FLIGHT_FEATURES].copy()
    y = sub[TARGET_REGRESSION].copy()
    return X, y


def get_model3_data(df: pd.DataFrame, min_class_count: int = 30):
    """Classification multi-classe du motif (famille_retard), restreinte
    aux vols en retard ET dont le motif a été renseigné dans NETLINE/AMOS.
    Les vols en retard sans motif documenté ('AUCUN') sont exclus: ce
    n'est pas une vraie classe métier mais un motif manquant — c'est
    d'ailleurs précisément le problème de fiabilité de saisie identifié
    dans le rapport de stage. Les classes trop rares sont fusionnées en
    'AUTRES' pour éviter des classes à 1-2 exemples lors du split."""
    sub = df[(df[TARGET_BINARY] == 1) & (df[TARGET_FAMILY] != "AUCUN")].copy()
    counts = sub[TARGET_FAMILY].value_counts()
    rare = counts[counts < min_class_count].index
    sub[TARGET_FAMILY] = sub[TARGET_FAMILY].replace({c: "AUTRES" for c in rare})

    features = PRE_FLIGHT_FEATURES + POST_DELAY_EXTRA_FEATURES
    X = sub[features].copy()
    y = sub[TARGET_FAMILY].copy()
    return X, y


CATEGORICAL_FEATURES_M3 = CATEGORICAL_FEATURES_PRE + ["tranche_retard"]
NUMERIC_FEATURES_M3 = NUMERIC_FEATURES_PRE + ["retard_min"]
