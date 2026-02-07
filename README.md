# Dashboard d'Analyse des Ventes - Streamlit & DuckDB

## Description du projet

Application web interactive développée avec **Streamlit** permettant l'analyse de données de ventes provenant de trois sources : Amazon, Burger King et McDonald's. Le dashboard utilise **DuckDB** pour les requêtes SQL et propose des visualisations interactives pour explorer les indicateurs clés de performance (KPI).

## Objectifs

- Téléverser et analyser des fichiers CSV de ventes
- Visualiser automatiquement les KPI (moyenne, minimum, maximum, nombre d'entrées)
- Filtrer dynamiquement les données par valeur, attribut ou item
- Générer des graphiques analytiques (distribution, tendances, répartition)
- Exporter les données filtrées au format CSV

## Technologies utilisées

- **Python 3.x**
- **Streamlit** : interface web interactive
- **DuckDB** : base de données SQL embarquée
- **Pandas** : manipulation de données
- **Matplotlib** : visualisations graphiques
- **Seaborn** : graphiques statistiques avancés

## Installation

### Prérequis
```bash
pip install streamlit pandas duckdb matplotlib seaborn
```

### Lancement de l'application
```bash
streamlit run code.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## Jeux de données supportés

### 1. Amazon Sales Dataset
- **Source** : Kaggle - Amazon Product Ratings and Reviews
- **Contenu** : 1000+ évaluations de produits Amazon
- **Colonnes clés** : `product_name`, `category`, `discounted_price`, `actual_price`, `rating`
- **Analyses** : distribution des prix, catégories populaires, tendances de notation

### 2. Burger King Dataset
- **Source** : Données financières 2021-2023
- **Contenu** : Performances financières et expansion
- **Colonnes clés** : `attribute`, `value`, `global_us_usc`
- **Analyses** : croissance des ventes, franchises vs. établissements gérés, expansion géographique

### 3. McDonald's Dataset
- **Source** : Données financières 2021-2023
- **Contenu** : Revenus et structure organisationnelle
- **Colonnes clés** : `table_name`, `heading`, `item`, `date`, `value`
- **Analyses** : évolution des revenus, modèle franchise vs. company-operated, récupération post-COVID

## Fonctionnalités principales

### Détection automatique du type de fichier
Le dashboard identifie automatiquement le type de dataset uploadé et adapte les visualisations en conséquence.

### Indicateurs clés de performance (KPI)
- Nombre d'entrées
- Moyenne des valeurs
- Valeur minimale
- Valeur maximale

### Visualisations interactives
1. **Distribution des valeurs** : histogramme montrant la répartition des prix/valeurs
2. **Violin Plot** : analyse détaillée de la densité de distribution
3. **Évolution cumulée** : courbe cumulative avec seuil 50%
4. **Top catégories/items** : barres horizontales des éléments dominants
5. **Analyse temporelle** : tendances année par année (BK/McDonald's uniquement)
6. **Répartition par type de revenus** : comparaison franchise vs. company-operated (McDonald's uniquement)

### Filtres dynamiques
- Filtre par plage de valeurs (slider)
- Filtre par attribut (Burger King)
- Filtre par item (McDonald's)
- Sélection du nombre de lignes à afficher

### Export de données
Téléchargement des données filtrées au format CSV.



## Guide d'utilisation

1. **Lancer l'application** : `streamlit run code.py`
2. **Uploader un fichier CSV** : cliquer sur "Browse files" et sélectionner un des trois datasets
3. **Explorer les KPI** : visualiser automatiquement les indicateurs clés
4. **Utiliser les filtres** : ajuster les plages de valeurs dans la barre latérale
5. **Analyser les graphiques** : lire les interprétations sous chaque visualisation
6. **Exporter les résultats** : télécharger les données filtrées si nécessaire

## Exemples d'insights

### Amazon
- Identification des catégories de produits les plus vendues
- Analyse de la distribution des prix avec remises
- Corrélation entre prix et notation

### Burger King & McDonald's
- Croissance année après année (2021-2023)
- Comparaison franchises vs. établissements gérés
- Tendances d'expansion géographique
- Impact de la récupération post-COVID

## Gestion des erreurs

Le dashboard gère automatiquement :
- Encodages multiples (UTF-8, UTF-8-BOM, Latin-1, CP1252)
- Valeurs manquantes ou aberrantes
- Formats de dates variables
- Colonnes numériques encodées en texte

## Notes techniques

- **Encodage** : détection automatique avec gestion du BOM UTF-8
- **Performance** : utilisation de DuckDB pour des requêtes SQL rapides
- **Responsive** : mise en page adaptative avec colonnes Streamlit
- **Cache** : fonction de conversion CSV mise en cache pour optimisation


## Licence

Ce projet a été réalisé dans le cadre d'une évaluation académique MBA ESG - Management Opérationnel.

---

**Développé avec** Python, Streamlit, DuckDB, Seaborn | **Année** 2026
