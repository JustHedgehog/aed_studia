import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import seaborn as sns
from sklearn_extra.cluster import KMedoids

#Zadanie 1

# Załadowanie danych
df = pd.read_csv("DATA.csv")

# Wyrzucenie z danych kolumn STUDENT ID oraz GRADE
X = df.iloc[:, 1:-1].values.astype(float)

# Standaryzacja danych za pomocą StandardScalera
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pobranie liczby od użytkownika
try:
    k = int(input("Ilość klastrów (domyślnie to 5): ") or 5)
except ValueError:
    k = 5
    print("Zły format wprowadzonych danych, użyto wartości domyślnej k=5")

#Zadanie 2
# K-Means (random init)
kmeans_km = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
labels_km = kmeans_km.fit_predict(X_scaled)
centroids_km = kmeans_km.cluster_centers_

# K-Means++
kmeans_kmpp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
labels_kmpp = kmeans_kmpp.fit_predict(X_scaled)
centroids_kmpp = kmeans_kmpp.cluster_centers_

# K-Medoids
kmdeoids = KMedoids(n_clusters=k, metric='euclidean', random_state=42)
labels_kmed = kmdeoids.fit_predict(X_scaled)
centroids_kmed = kmdeoids.cluster_centers_

# Dodanie etykiet do dataframe'a
df['KMEANS_Cluster'] = labels_km + 1
df['KMEANS_PP_Cluster'] = labels_kmpp + 1
df['KMEDOIDS_Cluster'] = labels_kmed + 1

# Zapis dataframe'a
df.to_csv('clustered_data.csv', index=False)
print("Dane zapisane do 'clustered_data.csv'")

# PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

# Kolory i markery
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown'][:k]
markers = ['o', 's', '^', 'D', '*', 'v', 'p', 'h', '<', '>'][:k]

# Wizualizacja
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
titles = ['K-Means', 'K-Means++', 'K-Medoids']
labels_list = [labels_km, labels_kmpp, labels_kmed]
centroids_list = [centroids_km, centroids_kmpp, centroids_kmed]

for idx, ax in enumerate(axs):
    # Plot points
    for i in range(k):
        mask = labels_list[idx] == i
        ax.scatter(pcs[mask, 0], pcs[mask, 1], c=colors[i], marker=markers[i], label=f'Cluster {i + 1}')

    # Project and plot centers
    cent_pcs = pca.transform(centroids_list[idx])
    ax.scatter(cent_pcs[:, 0], cent_pcs[:, 1], c='black', marker='X', s=200, label='Centers')

    ax.set_title(titles[idx])
    ax.legend()

plt.tight_layout()
plt.show()

#Zadanie 3

df = pd.read_csv("DATA.csv")
X_scaled = scaler.fit_transform(X)

# Agglomerative Clustering
agg_model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
labels_agg = agg_model.fit_predict(X_scaled)

# Dataframe dla Agglomerative
df_scaled = pd.DataFrame(X_scaled, index=df['STUDENT ID'], columns=df.columns[1:-1])

# DBScan z domyślnymi parametrami
db_model = DBSCAN()
labels_db = db_model.fit_predict(X_scaled)

# Przemapowanie etykiet DBSCAN, szum oznaczony jako 1 zamiast -1
unique_db = sorted(np.unique(labels_db))
label_map = {old: new for new, old in enumerate(unique_db, 1)}
labels_db_mapped = np.array([label_map[label] for label in labels_db])

# Dodanie etykiet do dataframe'u
df['Agglomerative_Cluster'] = labels_agg + 1
df['DBSCAN_Cluster'] = labels_db_mapped

# Zapis danych
df.to_csv('clustered_data_additional.csv', index=False)
print("Dane zapisane do 'clustered_data_additional.csv'")

# Agglomerative wizualizacja wraz z heatmapą
g = sns.clustermap(df_scaled, method='ward', metric='euclidean', cmap='viridis',
                   row_cluster=True, col_cluster=True, standard_scale=None)
g.fig.suptitle('Agglomerative Clustering')
plt.show()

# DBSCAN wizualizacja
plt.figure(figsize=(8, 6))
colors = ['cyan', 'red', 'blue', 'green', 'yellow', 'magenta', 'black', 'orange', 'purple', 'brown']
markers = ['o', 's', '^', 'D', '*', 'v', 'p', 'h', '<', '>']

num_clusters_db = len(unique_db)
for i, lbl in enumerate(unique_db):
    mask = labels_db == lbl
    label = f'Cluster {label_map[lbl]}' if lbl != -1 else 'Szum (1)'
    plt.scatter(pcs[mask, 0], pcs[mask, 1], c=colors[i % len(colors)], marker=markers[i % len(markers)], label=label)

plt.title('DBSCAN')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Zadanie 4

# Zakres
ks = range(2, 16)

scores_km = []
scores_kmpp = []
scores_kmed = []
scores_agg = []

for k in ks:
    # K-Means (random)
    kmeans_km = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    labels_km = kmeans_km.fit_predict(X_scaled)
    scores_km.append(silhouette_score(X_scaled, labels_km))

    # K-Means++
    kmeans_kmpp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels_kmpp = kmeans_kmpp.fit_predict(X_scaled)
    scores_kmpp.append(silhouette_score(X_scaled, labels_kmpp))

    # K-Medoids
    kmdeoids = KMedoids(n_clusters=k, metric='euclidean', random_state=42)
    labels_kmed = kmdeoids.fit_predict(X_scaled)
    scores_kmed.append(silhouette_score(X_scaled, labels_kmed))

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    labels_agg = agg.fit_predict(X_scaled)
    scores_agg.append(silhouette_score(X_scaled, labels_agg))

# Zanalezienie optymalnego K
optimal_km = ks[np.argmax(scores_km)]
optimal_kmpp = ks[np.argmax(scores_kmpp)]
optimal_kmed = ks[np.argmax(scores_kmed)]
optimal_agg = ks[np.argmax(scores_agg)]

print(f"Optymalne k dla K-Means: {optimal_km}")
print(f"Optymalne k dla K-Means++: {optimal_kmpp}")
print(f"Optymalne k dla K-Medoids: {optimal_kmed}")
print(f"Optymalne k dlda Agglomerative: {optimal_agg}")

# Plot wyników
plt.figure(figsize=(10, 6))
plt.plot(ks, scores_km, marker='o', label='K-Means')
plt.plot(ks, scores_kmpp, marker='s', label='K-Means++')
plt.plot(ks, scores_kmed, marker='^', label='K-Medoids')
plt.plot(ks, scores_agg, marker='D', label='Agglomerative')
plt.xlabel('Ilość klastrów (k)')
plt.ylabel("Średni wynik Silhouette'a")
plt.title('Wynik Silhouette vs. Ilość klastrów')
plt.legend()
plt.grid(True)
plt.show()