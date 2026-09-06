import pandas as pd

def clean_data(df, is_train=True):
    """preproccessing du dataframe """
    df = df.copy()
    
    # on garde la valeur numérique des stages et on convertit les grades en numérique
    df['T Stage'] = df['T Stage'].astype(str).str.extract(r'(\d+)').astype(int)
    df['N Stage'] = df['N Stage'].astype(str).str.extract(r'(\d+)').astype(int)
    # le grade de la tumeur est déjà dans un format numérique mais on a une valeur anormale " anaplastic; Grade IV" qui correspond à un grade 4
    grade_map = {'1': 1, '2': 2, '3': 3, ' anaplastic; Grade IV': 4, '4': 4}
    df['Grade'] = df['Grade'].map(grade_map)
    # on convertit les colonnes en numérique
    stage_6_map = {'IIA': 1, 'IIB': 2, 'IIIA': 3, 'IIIB': 4, 'IIIC': 5}
    df['6th Stage'] = df['6th Stage'].map(stage_6_map)
    
    # encodage binaire
    df['Estrogen Status'] = df['Estrogen Status'].map({'Positive': 1, 'Negative': 0})
    df['Progesterone Status'] = df['Progesterone Status'].map({'Positive': 1, 'Negative': 0})
    df['A Stage'] = df['A Stage'].map({'Localised': 0, 'Regional': 1, 'Distant': 2}) 
    
    # Encodage
    df = pd.get_dummies(df, columns=['Race', 'Marital Status', 'Differentiate'])
    
    # Séparation des features et des targets
    target_clf = None
    target_reg = None
    ids = df['Id']
    
    if is_train:
        target_clf = df['Status'].map({'Alive': 0, 'Dead': 1})
        target_reg = df['Survival Months']
        
    # On supprime les colonnes qui ne sont pas des features
    cols_to_drop = ['Id', 'Status', 'Survival Months']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return X, target_clf, target_reg, ids

def get_aligned_data(train_path, eval_path):
    """Charge et aligne les colonnes de train et eval après le get_dummies."""
    df_train = pd.read_csv(train_path)
    df_eval = pd.read_csv(eval_path)
    
    X_train, y_clf, y_reg, _ = clean_data(df_train, is_train=True)
    X_eval, _, _, eval_ids = clean_data(df_eval, is_train=False)
    
    # Alignement strict des colonnes (pour éviter les erreurs si une ethnie manque dans eval)
    X_train, X_eval = X_train.align(X_eval, join='left', axis=1, fill_value=0)
    
    return X_train, y_clf, y_reg, X_eval, eval_ids