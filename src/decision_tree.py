import pandas as pd
from prepocessing import get_aligned_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, y_clf, y_reg, X_eval, eval_ids = get_aligned_data('datasets/train.csv', 'datasets/eval.csv')

# 2. training
X_t, X_val, y_t, y_val = train_test_split(X_train, y_clf, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_t, y_t)

# 3. training final sur 100% des données et prediction sur eval.csv
clf_final = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_final.fit(X_train, y_clf)
eval_status_preds = clf_final.predict(X_eval)

mean_survival = y_reg.mean()
death_status = {0: 'Alive', 1: 'Dead'}

prediction = pd.DataFrame({
    'Id': eval_ids,
    'Survival Months': [mean_survival] * len(eval_ids), 
    'Status': [death_status[p] for p in eval_status_preds]
})

prediction.to_csv('datasets/predictions_tree.csv', index=False)
