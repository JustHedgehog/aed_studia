import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Zadanie 1 Pobieranie i ładowanie danych

X_train = pd.read_csv("data/Train/X_train.txt", sep='\s+', header=None).values
y_train = pd.read_csv("data/Train/y_train.txt", sep='\s+', header=None).values.ravel()
X_test = pd.read_csv("data/Test/X_test.txt", sep='\s+', header=None).values
y_test = pd.read_csv("data/Test/y_test.txt", sep='\s+', header=None).values.ravel()

activity_labels = pd.read_csv("data/activity_labels.txt", sep='\s+', header=None, index_col=0)[1].to_dict()

print("Dane załadowane pomyślnie.")

# Zadanie 2: Budowa modeli z domyślnymi parametrami
models = {
    'SVM': SVC(probability=True),
    'kNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Model {name} zbudowany.")

# Zadanie 3 Ocena modeli
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix for {name}:\n{cm}')
    print(classification_report(y_test, y_pred, target_names=[activity_labels[i] for i in sorted(activity_labels.keys())]))
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    print(f'{name} - ACC: {acc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

for name, model in models.items():
    evaluate_model(model, X_test, y_test, name)

# Zadanie 4 Wybór najlepszego algorytmu na podstawie CV
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = (scores.mean(), scores.std())
    print(f'{name}: Średnia ACC: {scores.mean():.4f}, Odchylenie std: {scores.std():.4f}')

best_algo = max(cv_results, key=lambda x: cv_results[x][0])
print(f'Najlepszy algorytm na podstawie CV: {best_algo}')

# Zadanie 5 Optymalizacja parametrów i testowanie
param_grids = {
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
    'kNN': {'n_neighbors': [3, 5, 7, 10]},
    'Decision Tree': {'max_depth': [None, 5, 10, 20]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
}

best_models = {}
best_scores = {}
for name in models:
    grid = GridSearchCV(models[name], param_grids[name], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    best_scores[name] = grid.best_score_
    print(f'{name} - Najlepsze parametry: {grid.best_params_}, Najlepszy wynik CV: {grid.best_score_:.4f}')

overall_best = max(best_scores, key=best_scores.get)
best_model = best_models[overall_best]
y_test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f'Najlepszy model: {overall_best}, Wynik na zbiorze testowym (ACC): {test_acc:.4f}')
evaluate_model(best_model, X_test, y_test, overall_best + ' (optymalizowany)')

# Zadanie 5 Wizualizacja
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

classes = sorted(activity_labels.keys())
colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
markers = ['o', '^', 's', 'p', '*', 'h', 'D', 'v', '<', '>', '1', '2']

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Podwykres 1: Rozkład próbek treningowych (etykiety prawdziwe)
axs[0].set_title('Rozkład próbek treningowych')
for i, cls in enumerate(classes):
    mask = y_train == cls
    axs[0].scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], color=colors[i], marker=markers[i], label=activity_labels[cls])
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Podwykres 2: Wynik trenowania modelu (predykcje na treningowym)
axs[1].set_title('Wynik trenowania modelu')
for i, cls in enumerate(classes):
    mask = y_train_pred == cls
    axs[1].scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], color=colors[i], marker=markers[i], label=activity_labels[cls])
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Podwykres 3: Wynik testowania modelu (predykcje na testowym) - zawiera też rozkład testowych poprzez predykcje
axs[2].set_title('Wynik testowania modelu')
for i, cls in enumerate(classes):
    mask = y_test_pred == cls
    axs[2].scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], color=colors[i], marker=markers[i], label=activity_labels[cls])
axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()