import pandas as pd
from prepocessing import get_aligned_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

X_train, y_clf, y_reg, X_eval, eval_ids = get_aligned_data('datasets/train.csv', 'datasets/eval.csv')

# 2. training
X_t, X_val, y_t, y_val = train_test_split(X_train, y_reg, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_t, y_t)

# 3. training final sur 100% des données et prediction sur eval.csv
reg_final = LinearRegression()
reg_final.fit(X_train, y_reg)
eval_months_preds = reg_final.predict(X_eval)

majority_class = y_clf.mode()[0] # Calcule la classe la plus fréquente (0: Alive)
status_map_rev = {0: 'Alive', 1: 'Dead'}

prediction = pd.DataFrame({
    'Id': eval_ids,
    'Survival Months': eval_months_preds,
    'Status': [status_map_rev[majority_class]] * len(eval_ids) 
})

prediction.to_csv('datasets/predictions_reg.csv', index=False)
