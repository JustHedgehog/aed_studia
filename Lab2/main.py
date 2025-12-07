import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Zadanie 1:")

# 1. Wymiar wczytanych danych
print("1. Wymiar wczytanych danych:", X.shape)

# 2. Ilość wartości unikatowych w wektorze target
print("2. Ilość wartości unikatowych w wektorze target:", len(np.unDique(y)))

# 3. Sprawdzenie, które kolumny zawierają najmniej informacji
mi = mutual_info_classif(X, y)
sorted_indices = np.argsort(mi)
print("3. Propozycja usunięcia kolumn o najmniejszej informacji (MI < 0.05):")
remove_features = []
keep_mask = []
for idx in sorted_indices:
    if mi[idx] < 0.05:
        remove_features.append(feature_names[idx])
    else:
        keep_mask.append(idx)
print("Usuwane kolumny:", remove_features)

# 4. Zapis w pliku dataset_cut.csv
X_cut = X[:, keep_mask]
cut_feature_names = feature_names[keep_mask]
df_cut = pd.DataFrame(X_cut, columns=cut_feature_names)
df_cut['TARGET'] = y
df_cut.to_csv('dataset_cut.csv', sep=';', index=False)

print("\nZadanie 2:")

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1-2. PCA to 5 dimensions
pca_5 = PCA(n_components=5, random_state=42)
X_pca_5 = pca_5.fit_transform(X_scaled)

# 3. Zapisz dataset_pca_5.csv
df_pca_5 = pd.DataFrame(X_pca_5, columns=['COMP1', 'COMP2', 'COMP3', 'COMP4', 'COMP5'])
df_pca_5['TARGET'] = y
df_pca_5.to_csv('dataset_pca_5.csv', sep=';', index=False)
print("3. Zapisano dataset_pca_5.csv")

# 4. Oblicz wariancję zbioru danych po redukcji
variances_5 = np.var(X_pca_5, axis=0)
print("4. Wariancje składowych po redukcji:", variances_5)

# 5. Explained variance ratio
print("5. Explained variance ratio: ", pca_5.explained_variance_ratio_)

print("6. Zsumowana warinacja: ", sum(pca_5.explained_variance_ratio_))

print("\nZadanie 3:")

# 1. optymalny wymiar
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
print(cum_var)
n_opt = np.where(cum_var >= 0.9)[0][0] + 1
print("1. Optymalny wymiar: ", n_opt, "ponieważ suma wariancji wyjaśnionej osiąga co najmniej 90%.")

# 2. Redukcja do n_opt
pca_n = PCA(n_components=n_opt)
X_pca_n = pca_n.fit_transform(X_scaled)

# 3. Zapis do dataset_pca_n.csv
comp_columns = [f'COMP{i+1}' for i in range(n_opt)]
df_pca_n = pd.DataFrame(X_pca_n, columns=comp_columns)
df_pca_n['TARGET'] = y
df_pca_n.to_csv('dataset_pca_n.csv', sep=';', index=False)
print("3. Zapisano dataset_pca_n.csv")

# 4. Explained variance ratio
print("4. Explained variance ratio:", pca_n.explained_variance_ratio_)

print("\nZadanie 4:")

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Zapis do pliku dataset_algorithm.csv
df_tsne = pd.DataFrame(X_tsne, columns=['COMP1', 'COMP2'])
df_tsne['TARGET'] = y
df_tsne.to_csv('dataset_algorithm.csv', sep=';', index=False)
print("Zapisano dataset_algorithm.csv (użyto t-SNE)")

# Wizualizacja
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1], label='Target (0: Malignant, 1: Benign)')
plt.title('2D t-SNE Visualization of Breast Cancer Dataset')
plt.xlabel('COMP1')
plt.ylabel('COMP2')
plt.grid(True)
plt.show()