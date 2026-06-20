# RAM Delay Intelligence

## Présentation générale

RAM Delay Intelligence est une application d’analyse et de prédiction de la ponctualité des vols, développée autour d’une problématique opérationnelle liée au suivi des retards, à la fiabilisation des motifs de retard et à l’aide à la décision.

Le projet combine Business Intelligence, analyse de données et Machine Learning afin de centraliser les informations de vol, produire des indicateurs de performance et proposer des modules d’aide à l’anticipation des retards.

L’application est construite avec Streamlit et organisée en plusieurs modules :

* Dashboard
* Analytics
* Weekly
* Performance Metrics
* Predict

Le projet s’inscrit dans le cadre d’un travail académique appliqué à une problématique de ponctualité des flottes aériennes.

---

## Contexte et problématique

Dans le cadre d’un projet académique en collaboration avec Royal Air Maroc, le travail s’est concentré sur l’analyse de la ponctualité des flottes aériennes et sur l’exploitation des données opérationnelles liées aux vols et aux retards.

La ponctualité constitue un indicateur majeur de performance dans le transport aérien. Les retards peuvent être liés à plusieurs facteurs : rotation des avions, contraintes de maintenance, contrôle aérien, correspondances, avaries, traitement avion ou encore contraintes de programmation.

La difficulté principale réside dans la consolidation et l’exploitation de données opérationnelles issues de plusieurs systèmes, notamment les données de suivi des vols et les données liées à la maintenance.

La problématique traitée est la suivante :

```text
Comment centraliser, fiabiliser et exploiter les données opérationnelles afin d'améliorer le suivi de la ponctualité des vols, analyser les causes de retard et anticiper les retards à venir ?
```

---

## Objectifs du projet

Le projet poursuit plusieurs objectifs :

* Nettoyer et structurer un jeu de données opérationnel relatif aux vols.
* Construire des variables exploitables pour l’analyse de la ponctualité.
* Produire des indicateurs de performance tels que le taux de retard, l’OTP strict et l’OTP15.
* Analyser les retards par famille de motif, sous-type avion, jour de semaine, période et route.
* Identifier les causes de retard les plus fréquentes et les périodes sensibles.
* Développer une application Streamlit interactive pour la visualisation et l’aide à la décision.
* Ajouter des modèles de Machine Learning pour estimer le risque de retard, la durée probable du retard et la famille de motif la plus probable.
* Fournir un outil exploitable pour le reporting opérationnel et l’analyse de performance.

---

## Données utilisées

Le projet utilise un jeu de données relatif à des segments de vols.

Le dataset contient :

```text
17 017 segments de vol
Période couverte : 29 mars 2025 au 24 juin 2025
Granularité : 1 ligne = 1 segment de vol
Nombre de sous-types avion : 14
Nombre d'immatriculations : 65
Nombre de routes : 353
```

Les données brutes sont stockées dans :

```text
data/RAM_raw.xlsx
```

Le dataset nettoyé et enrichi est stocké dans :

```text
data/RAM_clean.parquet
```

Les principales variables construites sont :

* retard en minutes ;
* indicateur de vol retardé ;
* indicateur de retard supérieur à 15 minutes ;
* tranche de retard ;
* famille de motif de retard ;
* période de la journée ;
* jour de semaine ;
* type de courrier ;
* route ;
* sous-type avion ;
* secteur d’origine ;
* secteur de destination.

---

## Architecture du projet

```text
Données brutes Excel
        |
        v
Nettoyage et feature engineering
        |
        v
Dataset propre au format Parquet
        |
        v
Analyse exploratoire et modules BI
        |
        v
Entraînement des modèles ML
        |
        v
Application Streamlit multi-pages
        |
        v
Dashboard, analyse, KPI et prédiction
```

Cette architecture permet de passer d’un fichier opérationnel brut à une application d’analyse interactive intégrant des indicateurs, des visualisations et des modèles prédictifs.

---

## Structure du dépôt

```text
.
├── app/
│   ├── app.py
│   ├── filters.py
│   ├── theme.py
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Analytics.py
│       ├── 3_Weekly.py
│       ├── 4_Performance_Metrics.py
│       └── 5_Predict.py
│
├── data/
│   ├── RAM_raw.xlsx
│   └── RAM_clean.parquet
│
├── models/
│   ├── model1_binary.joblib
│   ├── model2_regression.joblib
│   ├── model3_family.joblib
│   ├── model3_family.xgb.json
│   └── training_report.json
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── features.py
│   └── model.py
│
├── tests/
│   └── test_app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Fonctionnement du pipeline

### 1. Nettoyage et préparation des données

Le module suivant réalise le nettoyage et l’enrichissement du dataset :

```text
src/data_processing.py
```

Il permet de :

* charger le fichier Excel brut ;
* normaliser les noms de colonnes ;
* convertir les dates ;
* transformer les horaires de départ en minutes ;
* créer des tranches horaires ;
* construire les variables de retard ;
* regrouper les familles de motifs peu fréquentes ;
* créer les variables utilisées par l’application et les modèles ;
* exporter le dataset propre au format Parquet.

Commande :

```bash
python src/data_processing.py
```

Sortie générée :

```text
data/RAM_clean.parquet
```

---

### 2. Construction des features

Le module suivant définit les variables utilisées par les modèles :

```text
src/features.py
```

Les modèles sont séparés selon leur temporalité.

Variables pré-vol :

* sous-type avion ;
* compagnie opératrice ;
* type de courrier ;
* secteur d’origine ;
* secteur de destination ;
* jour de semaine ;
* week-end ou non ;
* mois ;
* période de la journée ;
* heure de départ programmée ;
* temps de vol programmé ;
* trafic interne ou externe.

Variables post-retard :

* durée du retard ;
* tranche de retard.

Cette séparation permet d’éviter la fuite de données : les modèles pré-vol n’utilisent que des informations connues avant le décollage.

---

### 3. Entraînement des modèles

Le module suivant entraîne les modèles :

```text
src/model.py
```

Commande :

```bash
python src/model.py
```

Trois modèles sont construits.

#### Modèle 1 : classification binaire

Objectif :

```text
Prédire si un vol sera à l'heure ou en retard.
```

Modèles comparés :

* Logistic Regression
* Random Forest
* XGBoost

Meilleur modèle obtenu :

```text
Logistic Regression
```

Performances sur le jeu de test :

```text
Accuracy : 0.895
F1-score : 0.901
Precision : 1.000
Recall : 0.820
ROC-AUC : 0.946
```

---

#### Modèle 2 : régression de la durée de retard

Objectif :

```text
Estimer la durée probable du retard pour les vols retardés.
```

Modèles comparés :

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

Meilleur modèle obtenu :

```text
Random Forest Regressor
```

Performances sur le jeu de test :

```text
MAE : 20.93 minutes
RMSE : 70.30 minutes
R² sur échelle logarithmique : 0.514
```

---

#### Modèle 3 : classification de la famille de motif

Objectif :

```text
Suggérer la famille de motif la plus probable lorsqu’un retard est déjà constaté.
```

Modèles comparés :

* Logistic Regression
* Random Forest
* XGBoost

Meilleur modèle obtenu :

```text
XGBoost
```

Performances sur le jeu de test :

```text
Accuracy : 0.559
F1 macro : 0.384
F1 pondéré : 0.536
```

Ce modèle est utilisé comme aide à la saisie ou à l’analyse. Il ne remplace pas la validation métier par un opérateur.

---

## Modules de l’application Streamlit

### Accueil

Fichier :

```text
app/app.py
```

La page d’accueil présente :

* le nombre total de vols analysés ;
* le taux de ponctualité ;
* le retard moyen des vols retardés ;
* la période couverte ;
* les différents modules disponibles.

---

### Dashboard

Fichier :

```text
app/pages/1_Dashboard.py
```

Ce module fournit une vue globale des retards.

Il permet d’analyser :

* le nombre de vols ;
* le nombre de vols retardés ;
* le taux de retard ;
* le retard moyen ;
* la répartition des retards par sous-type avion ;
* la sévérité des retards selon le seuil de 15 minutes ;
* la distribution des retards par tranche ;
* l’évolution quotidienne du taux de retard.

---

### Analytics

Fichier :

```text
app/pages/2_Analytics.py
```

Ce module analyse les causes de retard.

Il permet de suivre :

* les vols retardés ;
* les vols avec motif renseigné ;
* les vols sans motif renseigné ;
* la répartition par famille de codes ;
* le détail par famille de motif ;
* la distribution horaire des retards ;
* le retard moyen et médian par famille.

Ce module met aussi en évidence le problème de fiabilité de saisie lorsque certains retards ne disposent pas d’un motif documenté.

---

### Weekly

Fichier :

```text
app/pages/3_Weekly.py
```

Ce module permet d’analyser les retards par jour de semaine.

Il contient :

* un tableau croisé famille de motif × jour de semaine ;
* le taux de retard par jour ;
* le jour le plus problématique par famille de motif ;
* une heatmap des retards par famille et par jour.

---

### Performance Metrics

Fichier :

```text
app/pages/4_Performance_Metrics.py
```

Ce module calcule les indicateurs clés de ponctualité.

Indicateurs suivis :

* OTP strict : vols sans retard ;
* OTP15 : vols à l’heure ou avec retard inférieur ou égal à 15 minutes ;
* OTP ajusté : ponctualité corrigée en excluant certaines causes considérées comme externes ;
* évolution hebdomadaire du taux de ponctualité ;
* OTP par sous-type avion.

Sur le dataset utilisé, les indicateurs globaux sont :

```text
OTP strict : 41.51 %
OTP15 : 71.64 %
Taux de retard : 58.49 %
Retard moyen des vols retardés : 29.71 minutes
```

---

### Predict

Fichier :

```text
app/pages/5_Predict.py
```

Ce module ajoute une couche Machine Learning à l’application.

Il contient deux usages.

Avant le vol :

* estimation de la probabilité de retard ;
* estimation de la durée probable du retard.

Après constat du retard :

* suggestion de la famille de motif la plus probable ;
* affichage des probabilités par famille ;
* aide à la saisie et à l’analyse métier.

Les résultats doivent être interprétés comme des indicateurs d’aide à la décision, et non comme des décisions automatiques.

---

## Filtres interactifs

Les pages partagent un système de filtres permettant d’analyser les données selon plusieurs dimensions :

* sous-type avion ;
* immatriculation ;
* famille de motif de retard ;
* période temporelle.

Le module de filtres est défini dans :

```text
app/filters.py
```

---

## Design de l’application

Le style visuel de l’application est centralisé dans :

```text
app/theme.py
```

Il définit :

* une palette sobre inspirée d’un outil métier ;
* des couleurs principales ;
* des cartes KPI ;
* un bandeau d’en-tête ;
* une mise en forme homogène des pages ;
* une interface lisible pour l’analyse opérationnelle.

---

## Tests automatisés

Le projet contient des tests dans :

```text
tests/test_app.py
```

Ils vérifient notamment :

* le bon lancement des pages Streamlit ;
* le fonctionnement du module Predict ;
* l’absence de valeurs manquantes dans les colonnes cibles ;
* le chargement des modèles ;
* la compatibilité de la sauvegarde des modèles XGBoost ;
* l’absence de fuite de données liée à la variable route ;
* l’injection du thème sur les pages.

Commande :

```bash
pytest tests/test_app.py -v
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/your-username/ram-delay-intelligence.git
cd ram-delay-intelligence
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

Sous Windows :

```bash
venv\Scripts\activate
```

Sous Linux ou macOS :

```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Lancer le projet

### Option 1 : lancer directement l’application

Si le dataset propre et les modèles sont déjà présents :

```bash
streamlit run app/app.py
```

---

### Option 2 : reconstruire le pipeline complet

Nettoyer les données :

```bash
python src/data_processing.py
```

Réentraîner les modèles :

```bash
python src/model.py
```

Lancer l’application :

```bash
streamlit run app/app.py
```

---

## Dépendances principales

Les principales bibliothèques utilisées sont :

* pandas
* numpy
* scikit-learn
* xgboost
* streamlit
* plotly
* joblib
* openpyxl
* pyarrow
* packaging

---

## Résultats principaux

Sur le dataset fourni, le projet permet d’obtenir :

```text
Nombre de segments de vol : 17 017
Période : 29 mars 2025 au 24 juin 2025
Taux de retard : 58.49 %
OTP strict : 41.51 %
OTP15 : 71.64 %
Retard moyen des vols retardés : 29.71 minutes
```

Performances des modèles :

```text
Modèle 1 — Retard ou non
Meilleur modèle : Logistic Regression
F1-score : 0.901
ROC-AUC : 0.946

Modèle 2 — Durée du retard
Meilleur modèle : Random Forest Regressor
MAE : 20.93 minutes

Modèle 3 — Famille de motif
Meilleur modèle : XGBoost
Accuracy : 0.559
F1 pondéré : 0.536
```

Ces résultats montrent que le projet est utile pour l’analyse exploratoire, le reporting opérationnel et l’aide à la décision. Le modèle de classification des motifs reste volontairement présenté comme une aide à la saisie, car la prédiction exacte de la cause d’un retard demeure un problème complexe.

---

## Limites du projet

Le projet présente certaines limites :

* Les modèles pré-vol n’utilisent que des informations disponibles avant le décollage afin d’éviter la fuite de données.
* La prédiction de la durée du retard reste difficile en raison de la forte variabilité des retards longs.
* La classification de la famille de motif dépend fortement de la qualité de saisie des motifs dans les données.
* Certaines familles de retard sont plus rares que d’autres, ce qui rend la classification multiclasses plus complexe.
* Le modèle de diagnostic de motif ne remplace pas l’expertise opérationnelle.
* L’application est locale et n’est pas encore déployée sur un serveur de production.

---

## Améliorations possibles

Les améliorations futures peuvent inclure :

* Ajouter une analyse plus fine par route, escale et période de pointe.
* Intégrer davantage d’historique pour améliorer la robustesse des modèles.
* Tester des modèles temporels ou séquentiels pour mieux prendre en compte les effets de rotation.
* Ajouter une analyse d’importance des variables.
* Intégrer une base de données au lieu d’un fichier Parquet local.
* Déployer l’application sur une plateforme cloud.
* Ajouter un module d’export automatique de rapports.
* Améliorer le modèle de classification des familles de motifs avec des données mieux équilibrées.
* Ajouter des alertes lorsque certains KPI de ponctualité se dégradent.

---

## Compétences mobilisées

Ce projet met en évidence des compétences en :

* analyse de données ;
* Business Intelligence ;
* data cleaning ;
* feature engineering ;
* visualisation de données ;
* dashboarding avec Streamlit ;
* calcul de KPI opérationnels ;
* machine learning supervisé ;
* classification binaire ;
* régression ;
* classification multiclasses ;
* interprétation de résultats ;
* aide à la décision ;
* compréhension d’une problématique métier aérienne.

---

## Valeur académique et professionnelle

Ce projet illustre une démarche complète de data science appliquée à une problématique opérationnelle réelle.

Il montre comment passer d’un fichier brut à un outil d’analyse interactif combinant :

* nettoyage de données ;
* structuration des variables ;
* indicateurs de performance ;
* visualisations interactives ;
* modèles prédictifs ;
* aide à la décision.

Il est cohérent avec un profil orienté data science appliquée, pilotage de performance, supervision opérationnelle, machine learning et systèmes d’aide à la décision.

---

## Auteur

Zakaria Es-Salmy

Projet académique appliqué à l’analyse de la ponctualité des vols et à l’aide à la décision opérationnelle.
#   R A M _ d e l a y _ i n t e l l i g e n c e  
 