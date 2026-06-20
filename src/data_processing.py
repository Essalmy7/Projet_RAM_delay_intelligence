"""
Module de chargement, nettoyage et feature engineering du dataset RAM.

Source: extraction NETLINE/AMOS consolidée, 17017 vols (29 mars - 24 juin 2025).
Granularité: 1 ligne = 1 segment de vol (leg).
"""

import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = str(_ROOT / "data" / "RAM_raw.xlsx")
CLEAN_PATH = str(_ROOT / "data" / "RAM_clean.parquet")

# Familles de motifs de retard à conserver telles quelles (>100 occurrences
# parmi les vols réellement retardés). Le reste est regroupé en "AUTRES".
TOP_FAMILIES = [
    "ATC", "CORRESPONDANCE", "ROTATION", "TECHNIQIE",
    "AVARIE", "TIERS 2", "PROGRAMME", "TRAITEMENT AVION",
]

DAY_ORDER_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def _time_to_minutes(t):
    """Convertit une chaine 'HH:MM' ou 'HH:MM:SS' en minutes depuis minuit. NaN si invalide."""
    if pd.isna(t):
        return np.nan
    try:
        parts = str(t).split(":")
        h, m = int(parts[0]), int(parts[1])
        return h * 60 + m
    except (ValueError, IndexError):
        return np.nan


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset brut et construit les features utilisées par les
    modules BI et les modèles ML. Ne supprime aucune ligne (le dataset
    n'a pas de valeurs manquantes réelles, seulement des placeholders
    metier '-' / '0' qui sont gérés explicitement).
    """
    df = df.copy()

    # --- Normalisation des noms de colonnes (espaces parasites) ---
    df.columns = [c.strip() for c in df.columns]

    # --- Typage des dates ---
    df["DAY_OF_ORIGIN"] = pd.to_datetime(df["DAY_OF_ORIGIN"])
    df["DEP_DAY_SCHED"] = pd.to_datetime(df["DEP_DAY_SCHED"])

    # --- Heure programmée de départ, en minutes depuis minuit + tranche horaire ---
    df["dep_time_sched_min"] = df["DEP_TIME_SCHED"].apply(_time_to_minutes)
    df["dep_hour_sched"] = (df["dep_time_sched_min"] // 60).fillna(-1).astype(int)

    def hour_bucket(h):
        if h < 0:
            return "Inconnu"
        if 0 <= h < 6:
            return "Nuit (00h-06h)"
        if 6 <= h < 12:
            return "Matin (06h-12h)"
        if 12 <= h < 18:
            return "Après-midi (12h-18h)"
        return "Soir (18h-24h)"

    df["periode_journee"] = df["dep_hour_sched"].apply(hour_bucket)

    # --- Jour de semaine ordonné ---
    df["jour_semaine"] = pd.Categorical(
        df["DAY_OF_WEEK_DEP"].str.lower().str.strip(),
        categories=DAY_ORDER_FR,
        ordered=True,
    )
    df["is_weekend"] = df["jour_semaine"].isin(["samedi", "dimanche"])

    # --- Mois ---
    df["mois"] = df["Month DEP DAY SCHED"].str.strip()

    # --- Cible retard ---
    df["retard_min"] = df["Retard en min"].astype(float)
    df["is_delayed"] = (df["retard_min"] > 0).astype(int)
    df["is_delayed_15"] = (df["retard_min"] > 15).astype(int)

    # --- Famille de motif, regroupée et nettoyée ---
    fam = df["FAMILLE_DR_FUSION"].fillna("-").str.strip()
    df["famille_retard"] = np.where(
        fam.isin(TOP_FAMILIES), fam,
        np.where(fam == "-", "AUCUN", "AUTRES")
    )

    # --- Variables avion / route ---
    df["sous_type_avion"] = df["AC_SUBTYPE"].str.strip()
    df["immatriculation"] = df["AC_REGISTRATION"].str.strip()
    df["compagnie_operatrice"] = df["AT_RXP_AFFRT"].str.strip()
    df["secteur_origine"] = df["SECTEUR_ORIGIN"].str.strip()
    df["secteur_destination"] = df["SECTEUR_DESTINATION"].str.strip()
    df["route"] = df["O/D"].str.strip()
    df["aeroport_dep"] = df["DEP_AP_SCHED"].str.strip()
    df["aeroport_arr"] = df["ARR_AP_SCHED"].str.strip()
    df["type_courrier"] = df["MC_LC_CC"].str.strip()  # MC/LC/CC
    df["trafic_interne_externe"] = df["Interne/Externe"].replace("-", "Non renseigné")

    # --- Temps de vol programmé (minutes) ---
    df["temps_vol_sched_min"] = pd.to_numeric(
        df["Temps de vol programmé sur NETLINE"], errors="coerce"
    )

    # --- Tranche de retard (pour affichage BI, reprend la logique du rapport original) ---
    def tranche(r):
        if r <= 0:
            return "À l'heure"
        if r <= 15:
            return "0-15 min"
        if r <= 60:
            return "16-60 min"
        if r <= 120:
            return "1h-2h"
        if r <= 240:
            return "2h-4h"
        return "> 4h"

    df["tranche_retard"] = df["retard_min"].apply(tranche)

    return df


def _coerce_object_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Les colonnes 'object' à types mixtes (ex: int et str dans la même
    colonne) font échouer la sérialisation parquet. On les force en str."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str)
    return df


def build_clean_dataset(raw_path: str = RAW_PATH, out_path: str = CLEAN_PATH) -> pd.DataFrame:
    df_raw = load_raw(raw_path)
    df_clean = clean_and_engineer(df_raw)
    df_clean = _coerce_object_columns_to_string(df_clean)
    df_clean.to_parquet(out_path, index=False)
    return df_clean


def load_clean(path: str = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = build_clean_dataset()
    print(f"Dataset nettoyé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Sauvegardé dans {CLEAN_PATH}")
