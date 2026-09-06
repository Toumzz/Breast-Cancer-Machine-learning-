import pandas as pd
from prepocessing import get_aligned_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# 1. Chargement et alignement des données
# get_aligned_data retourne :
# - X_train : features d'entraînement
# - y_clf : cible de classification (statut de survie)
# - y_reg : cible de régression (nombre de mois de survie)
# - X_eval : features du jeu d'évaluation
# - eval_ids : identifiants des patients du jeu d'évaluation
X_train, y_clf, y_reg, X_eval, eval_ids = get_aligned_data('datasets/train.csv', 'datasets/eval.csv')

# 2. Séparation du jeu d'entraînement en train / validation
# On conserve 85% des données pour l'entraînement et 15% pour la validation.
X_t, X_val, y_clf_t, y_clf_val, y_reg_t, y_reg_val = train_test_split(
    X_train, y_clf, y_reg, test_size=0.15, random_state=42
)

# 3. Entraînement du modèle de régression sur les données d'entraînement
# Ce modèle prédit le nombre de mois de survie.
reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
reg.fit(X_t, y_reg_t)

# 4. Création des nouvelles features pour le classifieur
# On ajoute la prédiction de survie en mois comme feature au modèle de classification.
X_t_clf = X_t.copy()
X_t_clf['Predicted_Months'] = reg.predict(X_t)
X_val_clf = X_val.copy()
X_val_clf['Predicted_Months'] = reg.predict(X_val)

# 5. Entraînement du classifieur sur le jeu d'entraînement enrichi
# Le classifieur apprend à prédire le statut de survie (Alive/Dead).
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced_subsample',
    random_state=42
)
clf.fit(X_t_clf, y_clf_t)

# 6. Entraînement final sur 100% des données d'entraînement
# Après validation, on réentraine le régressseur sur tout le jeu d'entraînement disponible.
reg_final = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
reg_final.fit(X_train, y_reg)

# 7. Prédiction du nombre de mois de survie pour le jeu d'évaluation
eval_months_preds = reg_final.predict(X_eval)

# 8. Préparation des données d'entrée pour le classifieur final
# On ajoute la prédiction de mois sur les données d'entraînement et d'évaluation.
X_train_final_clf = X_train.copy()
X_train_final_clf['Predicted_Months'] = reg_final.predict(X_train)

X_eval_final_clf = X_eval.copy()
X_eval_final_clf['Predicted_Months'] = eval_months_preds

# 9. Entraînement complet du classifieur sur toutes les données d'entraînement
clf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced_subsample',
    random_state=42
)
clf_final.fit(X_train_final_clf, y_clf)

# 10. Prédiction du statut de survie pour le jeu d'évaluation
eval_status_preds = clf_final.predict(X_eval_final_clf)

# 11. Préparation du mappage des classes vers un label lisible
status_map_rev = {0: 'Alive', 1: 'Dead'}

# 12. Création du fichier de soumission
prediction = pd.DataFrame({
    'Id': eval_ids,
    'Survival Months': eval_months_preds,
    'Status': [status_map_rev[p] for p in eval_status_preds]
})

prediction.to_csv('datasets/predictions_mon_model.csv', index=False)
