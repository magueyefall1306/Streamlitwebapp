# MBA-2---Python
 Projet : Exposition de données sous la forme d’une API
 # Dashboard KPI – Streamlit & DuckDB

## Description
Application web interactive développée avec Streamlit permettant
d’analyser des données de ventes à partir de fichiers CSV.
Les données sont stockées et interrogées via DuckDB, une base de données
analytique embarquée, offrant des requêtes SQL rapides et efficaces sans
dépendre d’un serveur externe

## Fonctionnalités

Téléversement de fichiers CSV

Stockage des données dans DuckDB (en mémoire)

Calcul de 4 indicateurs clés de performance (KPI) :

Nombre de ventes
Valeur moyenne
Valeur minimale
Valeur maximale

Visualisation de chaque KPI via un graphique distinct :

Histogramme
Boxplot
Courbe cumulative
Diagramme circulaire (camembert)

Filtres dynamiques via curseur (adapté au fichier chargé)
Mise à jour automatique des KPI et des visualisations
Interface web interactive sur une seule page

## Technologies utilisées
- Python 3.11
- Streamlit
- DuckDB
- Pandas
- Matplotlib


## Lancer l’application

pip install -r requirements.txt  qui sert à installer automatiquement toutes les bibliothèques nécessaires au projet, en une seule fois.

streamlit run app.py
