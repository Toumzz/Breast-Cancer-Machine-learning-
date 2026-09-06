import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/train.csv')

df = df.drop(columns=['Age', 'Id'])

dictionnaire_grades = {
    '1': 1,
    '2': 2,
    '3': 3,
    ' anaplastic; Grade IV': 4  
}

# Convertir T Stage et N Stage en int
df['T Stage'] = pd.to_numeric(df['T Stage'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce').astype(int)
df['N Stage'] = pd.to_numeric(df['N Stage'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce').astype(int)

df['Grade'] = df['Grade'].replace(dictionnaire_grades)
df['Grade'] = pd.to_numeric(df['Grade'])

# Convertir les colonnes texte/booléennes
df['Estrogen Status'] = (df['Estrogen Status'] == 'Positive').astype(int)
df['Progesterone Status'] = (df['Progesterone Status'] == 'Positive').astype(int)
df['Status'] = (df['Status'] == 'Dead').astype(int)

# One-hot encoding pour Marital Status
#df = pd.get_dummies(df, columns=['Race'], prefix='Race', dtype=int)
#df = pd.get_dummies(df, columns=['Marital Status'], prefix='Marital_Status', dtype=int)
#df = pd.get_dummies(df, columns=['Differentiate'], prefix='Differentiate', dtype=int)
#j'hésite à inclure le differentiate et la marital status car peu de relevance dans la corrélation
#je n'inclu pas l'age et l'éthnie car aucun corrélation et on a pas plus d'info concernant le contexte socio-économique du patient

# Calculer la corrélation
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.tight_layout()
plt.show()

df.to_csv('datasets/correlation.csv', index=False)