## Projet PLDAC : Apprentissage à base de modèles de langage sur des données non-linguistiques

---

### Introduction

Nous connaissons le fonctionnement des modèles classiques pour l'apprentissage supervisé qui consiste à prédire un résultat selon les caractéristiques données. En entrée, nous avons une base de données X, que nous appelons "dataset", contenant des exemples ayant des valeurs selon leurs caractéristiques, que nous appelons "features", ainsi que les résultats y, que nous appelons "label", qui permettent au modèle de faire de la prédiction. Le modèle peut résoudre des problèmes de régression ou de classification selon la nature de y, plus précisément, s'il s'agit de valeurs continues ou discrètes.</br>

Dans ce projet intitulé "Apprentissage à base de modèles de langage sur des données non-linguistiques", nous avons pour objectif de découvrir si un modèle de langage est capable de réaliser des prédictions sur des données numériques. Habituellement, un modèle de langage effectue le traitement de langage naturel qui permet de traiter du texte et non un tableau de données. La question suivante se pose : Les modèles de langages, capables de traiter des données textuelles, peuvent-ils également traiter des données numériques ? Si oui, comment le font-ils ? </br>

Nous allons nous focaliser sur la résolution de problèmes de classifications en priorité qui permettra de mieux visualiser les prédictions que les problèmes de régression. Nous avons donc en entrée, un dataset de textes sur lequel le modèle effectue le fine-tuning sans les labels. Le but de ce projet consiste surtout à comprendre les méthodes d’un modèle de langage pour la prédiction, ce qui correspond à la partie “Explainability” du poster de LIFT. </br>

---

### Contenu

Nous avons deux fichiers : </br>
- projet.ipynb : Fichier avec quelques expérimentations et résultats
- projet-ft.ipynb : Fichier à exécuter de préférence sur Google Colab pour faciliter l'entraînement des modèles de langage

---

- Réalisé par Salwa MUJAHID
- Projet dirigé par Christophe MARSALA
- DAC 2023-2024 Sorbonne Université
