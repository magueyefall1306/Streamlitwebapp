# Dashboard KPI – DuckDB

Ce projet est une **application Streamlit** pour l'analyse et la visualisation des ventes à partir de fichiers CSV. Elle utilise **DuckDB** pour les requêtes SQL en mémoire et **matplotlib / seaborn** pour les visualisations.  

L'application est conçue pour gérer différents types de fichiers : Amazon, Burger King, McDonald's, ou d'autres fichiers CSV génériques.

---

## Fonctionnalités

- Téléversement et lecture robuste de fichiers CSV (gestion d'encodage UTF-8, Latin-1, CP1252 et BOM).  
- Détection automatique du type de fichier : Amazon, Burger King, McDonald's ou inconnu.  
- Nettoyage spécifique des données pour Amazon (prix, pourcentage de réduction).  
- Filtres interactifs dans la sidebar :  
  - Filtre par valeur (`discounted_price`, `value`, etc.)  
  - Filtre par attribut (pour BK) ou item (pour McDonald's)  
- KPI dynamiques :  
  - Nombre d'entrées  
  - Moyenne, minimum et maximum des valeurs  
- Visualisations interactives :  
  - Histogramme  
  - Violin plot  
  - Courbe cumulative  
  - Bar chart top N pour catégories/items  
- Analyses temporelles et par type de revenus (pour BK et McDonald's).  
- Statistiques descriptives et métriques supplémentaires (médiane, écart-type, variance, coefficient de variation).  
- Aperçu des données filtrées et téléchargement en CSV.

---

## Installation

1. **Cloner le dépôt GitHub**  
```bash
git clone (https://github.com/magueyefall1306/Streamlitwebapp.git
cd Streamlitwebapp
