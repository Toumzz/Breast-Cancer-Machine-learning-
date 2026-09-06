import pandas as pd
from prepocessing import get_aligned_data
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

'''
    UTILISER POUR LE DATAMINING (N-2) :
'''

# 1. Chargement des données
X_train, y_clf, y_reg, _, _ = get_aligned_data('datasets/train.csv', 'datasets/eval.csv')


# ==========================================
# ETAPE 0 : Standardisation des données
# ==========================================
# mettres les variables à la même échelle
# variables à la même échelle (ex: Taille de la tumeur vs Age/Grade)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# on fait 3 vluster
N_CLUSTERS = 3


# ==========================================
# ETAPE 1 : K-Means
# ==========================================
#trainging de k means
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# Calcul du score de silhouette (mesure la qualité des clusters plus c'est proche de 1 mieux c'est)
sil_kmeans = silhouette_score(X_scaled, clusters_kmeans)
print(f"Silhouette (K-Means) : {sil_kmeans:.3f}")


# ==========================================
# ETAPE 2 : Clustering Hiérarchique
# ==========================================
#training du modele hierarchique
# 'ward' minimise la variance au sein des clusters
hc = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
clusters_hc = hc.fit_predict(X_scaled)

sil_hc = silhouette_score(X_scaled, clusters_hc)
print(f"Silhouette (Hiérarchique) : {sil_hc:.3f}")


# ==========================================
# ETAPE 3 : Data Mining 
# ==========================================
print("\n--- Analyse Data Mining (Basée sur le K-Means) ---")
#analyse en mettant els clusters dans le dataframe pour voir les profils de chaque cluster
df_analysis = X_train.copy()
df_analysis['Cluster'] = clusters_kmeans
df_analysis['Survival_Months_Réel'] = y_reg
df_analysis['Est_Décédée'] = y_clf

# On regroupe par cluster pour voir les moyennes
profils = df_analysis.groupby('Cluster').mean()

# print des stats pour chaque cluster
for i in range(N_CLUSTERS):
    print(f"\nProfil du Cluster {i} ({sum(clusters_kmeans == i)} patientes) :")
    print(f"- Espérance de vie moyenne : {profils.loc[i, 'Survival_Months_Réel']:.1f} mois")
    print(f"- Taux de mortalité      : {profils.loc[i, 'Est_Décédée']*100:.1f} %")
    
    # Si la variable 'Tumor Size' a été gardée dans votre preprocessing :
    if 'Tumor Size' in profils.columns:
        print(f"- Taille moy. tumeur     : {profils.loc[i, 'Tumor Size']:.1f} mm")
    
    # Si la variable 'Regional Node Positive' est présente :
    if 'Regional Node Positive' in profils.columns:
        print(f"- Ganglions positifs moy.: {profils.loc[i, 'Regional Node Positive']:.1f}")
