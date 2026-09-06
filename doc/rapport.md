# Projet 4DEV4RI : Le Cancer du Sein - Machine Learning & Data Mining

PAUL TOUMAZET 
61070 
C213 

---

## 1. Introduction
Ce projet fais l'objet d'une analyse d'un dataset sur le cancer du sein. L'objectif du projet et de faure du machine learning et du data mining sur ce dataset 

La difficulé de ce projet réside dans les colonnes $N-1$ et $N$. On doit prédire si la patiente est vivante ou décédée (`Status`, colonne $N$), sauf qu'on n'a pas accès au nombre de mois de survie (`Survival Months`, colonne $N-1$) dans le fichier d'évaluation. 


---


## 2. Exploration des données et Corrélations
Avant de faire le preprocessing, je fais une analyse de comment nos variables interagissaient entre elles. Pour ça, matrice de corrélation (Heatmap) via seabron (`correlation.py`). 

J'ai fais cette étape pour la compréhension et justifier mon preprocesing:

*   **Le lien évident :** La corrélation a tout de suite mis en évidence un lien fort entre `Survival Months` et `Status`. C'est ce qui nous a confortés dans notre idée de faire un modèle en cascade ($N-1 \to N$).
*   **Les "fausses" bonnes variables (Âge et Ethnie) :** On aurait pu penser que l'âge (`Age`) ou la race (`Race`) auraient un impact énorme sur la survie or, la matrice de corrélation a montré que ce n'était pas vraiment le cas ici. Sans données supplémentaires sur le contexte socio-économique des patientes (qui est souvent la vraie cause des inégalités de survie).
*   **Les vrais indicateurs de danger :** La corrélation a confirmé que les variables cliniques liées aux ganglions (`Regional Node Positive`), ainsi que les `T Stage` et `N Stage` étaient les véritables moteurs de la prédiction du statut.

---

## 3. Preprocessing : Nettoyage et Préparation
On sait qu'on ne peut pas donner des données brutes à un modèle. On a donc fait pas mal de nettoyage sur `train.csv`:

*   **Extraction des chiffres :** Les colonnes comme `T Stage` ou `N Stage` contenaient du texte (ex: "T1"). On a utilisé des expressions régulières pour ne garder que l'entier.
*   **La petite anomalie :** Dans la colonne `Grade`, en plus des "1, 2, 3", on est tombés sur une valeur bizarre : `" anaplastic; Grade IV"`. On l'a repérée et transformée en `4` pour garder une logique mathématique.
*   **Encodage ordinal :** Le stade `6th Stage` suit une évolution logique (IIA < IIB < IIIA...). On a donc fait un mapping manuel (1, 2, 3...).
*   **One-Hot Encoding :** Pour tout le reste (`Race`, `Marital Status`...), qui n'a pas d'ordre particulier, on a utilisé un `pd.get_dummies()`.
*   **Encodage des cibles en binaire:** "Positive/Negative" et "Alive/Dead" sont devenus des 1 et des 0.

---

## 4. Les Modèles testés

### Modèle 1 : Arbre de décision simple
On a commencé par un `DecisionTreeClassifier`. 
*   **Le plus :** C'est super visuel. En affichant l'arbre, on a tout de suite vu que la variable `Regional Node Positive` (le nombre de ganglions touchés) était la racine de l'arbre, donc la variable la plus importante.
*   **Le moins :** En termes de prédiction, le modèle avait un peu de mal avec les cas complexes et ne prenait pas en compte la notion de temps.

### Modèle 2 : La Régression "Naïve"
Le modèle de régression linéaire classique pour deviner `Survival Months`. 
Catastrophe au début, le score $R^2$ était  négatif... Le cancer est une maladie complexe, une simple ligne droite mathématique n'arrive pas à capter l'espérance de vie.

### Modèle 3 : Notre solution finale (La Cascade)

1.  On utilise un **Gradient Boosting Regressor** (plus robuste que la régression linéaire) pour deviner les mois de survie ($N-1$).
2.  On prend cette prédiction, on l'ajoute comme une nouvelle colonne dans notre dataset, et on donne le tout à un **Random Forest Classifier** pour deviner le `Status`.
*Note : on a ajouté le paramètre `class_weight='balanced_subsample'` car notre dataset est très déséquilibré (beaucoup plus de gens en vie que décédés) 85/15.*

---

## 5. Benchmarking : Nos résultats face au fichier "Naïf"

Comparaison des 3 modèles

| Métrique | Regression | Mon modèle |  Decision Tree| Modèle "Naif"|
| :--- | :--- | :--- | :--- | :--- |
| **$R^2$ (Mois de survie)** | 0.048 | **0.030** | -0.000006|  0.0
| **Accuracy (Exactitude)**| **0.849** | 0.827   | 0.86| 0.849
| **Precision** | 0.424 | **0.660** | 0.775| 0.424
| **Recall (Rappel)** | 0.5| **0.654** | 0.565|  	0.5

**Analyse des résultats :**

Le poitn important, elle est sur le **Recall**. On passe de 0.5 à **0.654**. Ça veut dire qu'on arrive à détecter environ 15% de décès supplémentaires par rapport au hasard. En médecine, c'est le plus important : on préfère faire quelques fausses alertes(faux positif, Precision à 66%) plutôt que de rater un patient dont la vie est en danger.

 $R^2$ pour l'estimation des mois est très petit (0.031), mais l'essentiel c'est qu'il soit **positif**. Ça prouve mathématiquement que notre étape  arrive à dégager une tendance et que ce n'est pas juste du hasard.

---

## 6. Data Mining : Apprentissage non supervisé (Clustering)
Pour le cluster avec **K-Means** (avec 3 clusters, après un bon `StandardScaler`). 

En analysant la moyenne des variables pour chaque cluster créé, on a remarqué 3 profils distincts :

*   **Cluster 0 (Les cas à risque faible) :** On y retrouve surtout des petites tumeurs (T1/T2) et quasiment aucun ganglion touché. Sans surprise, quand on vérifie a posteriori, c'est le groupe où la survie est la plus longue.
*   **Cluster 1 (Les cas intermédiaires) :** Des tumeurs un peu plus grosses, avec des statuts hormonaux assez variés.
*   **Cluster 2 (Les cas critiques) :** Ce cluster a regroupé presque tous les stades avancés 

Pour le cluster en **Hiérarchique** 

## 7. Conclusion
Ce projet apprend pas mal de choses concrètes :
1. L'Accuracy est trompeuse quand les données sont déséquilibrées. Ici dans un usage 'médical' le recall est le plus important.
2. L'attribut `Regional Node Positive` (ganglions) est de loin le critère qui a le plus de poids dans la survie.
3. Prédire directement l'état d'un patient sans prendre en compte la variable "temps" (l'espérance de survie) limite énormément les algorithmes. Le principe de calcule de $N-1$ puis $N$ à bien mieux fonctionné






