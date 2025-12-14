import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import mode

# Zadanie 1 Pobieranie i ładowanie danych

X_train = pd.read_csv("data/Train/X_train.txt", sep='\s+', header=None).values
y_train = pd.read_csv("data/Train/y_train.txt", sep='\s+', header=None).values.ravel()
X_test = pd.read_csv("data/Test/X_test.txt", sep='\s+', header=None).values
y_test = pd.read_csv("data/Test/y_test.txt", sep='\s+', header=None).values.ravel()

activity_labels = pd.read_csv("data/activity_labels.txt", sep='\s+', header=None, index_col=0)[1].to_dict()

print("Dane załadowane pomyślnie.")

# Dla XGBoost, etykiety od 0
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1

# Funkcja do oceny modelu
def evaluate_ensemble(y_true, y_pred, y_prob=None, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
    else:
        auc = None
    return acc, recall, f1, auc

# Zadanie 2 Uczenie zespołowe - Voting na całym zbiorze
clf1 = RandomForestClassifier()
clf2 = SVC()
clf3 = KNeighborsClassifier()

ensemble = VotingClassifier(estimators=[('rf', clf1), ('svm', clf2), ('knn', clf3)], voting='hard')

# Kroswalidacja
scoring = {'acc': 'accuracy', 'recall': 'recall_macro', 'f1': 'f1_macro'}
cv_results = cross_validate(ensemble, X_train, y_train, cv=5, scoring=scoring)

cv_df = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1'],
    'Mean': [cv_results['test_acc'].mean(), cv_results['test_recall'].mean(), cv_results['test_f1'].mean()],
    'Std': [cv_results['test_acc'].std(), cv_results['test_recall'].std(), cv_results['test_f1'].std()]
})

# Testowanie
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
test_acc, test_recall, test_f1, test_auc  = evaluate_ensemble(y_test, y_pred)

test_df = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1', 'AUC'],
    'Value': [test_acc, test_recall, test_f1, test_auc]
})

with pd.ExcelWriter('ensemble_learning.xlsx') as writer:
    cv_df.to_excel(writer, sheet_name='CV Results', index=False)
    test_df.to_excel(writer, sheet_name='Test Results', index=False)

print("Zadanie 2 ukończone.")

# Zadanie 3 Agregacja - na losowych podzbiorach

# Kroswalidacja - symulujemy CV
from sklearn.base import BaseEstimator

class AggregatingEnsemble(BaseEstimator):
    def __init__(self, clfs):
        self.clfs = clfs

    def fit(self, X, y):
        self.trained_clfs_ = []
        for clf in self.clfs:
            X_sub, _, y_sub, _ = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(1000))
            clf.fit(X_sub, y_sub)
            self.trained_clfs_.append(clf)
        return self

    def predict(self, X):
        preds = np.array([clf.predict(X) for clf in self.trained_clfs_])
        return mode(preds, axis=0)[0].ravel()

agg_ensemble = AggregatingEnsemble([RandomForestClassifier(), SVC(), KNeighborsClassifier()])

cv_results_agg = cross_validate(agg_ensemble, X_train, y_train, cv=5, scoring=scoring)

cv_df_agg = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1'],
    'Mean': [cv_results_agg['test_acc'].mean(), cv_results_agg['test_recall'].mean(), cv_results_agg['test_f1'].mean()],
    'Std': [cv_results_agg['test_acc'].std(), cv_results_agg['test_recall'].std(), cv_results_agg['test_f1'].std()]
})

# Testowanie
agg_ensemble.fit(X_train, y_train)
y_pred_agg = agg_ensemble.predict(X_test)
test_acc_agg, test_recall_agg, test_f1_agg, test_auc_agg = evaluate_ensemble(y_test, y_pred_agg)

test_df_agg = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1', 'AUC'],
    'Value': [test_acc_agg, test_recall_agg, test_f1_agg, test_auc_agg]
})

with pd.ExcelWriter('aggregating.xlsx') as writer:
    cv_df_agg.to_excel(writer, sheet_name='CV Results', index=False)
    test_df_agg.to_excel(writer, sheet_name='Test Results', index=False)

print("Zadanie 3 ukończone.")

# Zadanie 4 Boosting
# ADA Boost
ada = AdaBoostClassifier()

scoring = {'acc': 'accuracy', 'recall': 'recall_macro', 'f1': 'f1_macro', 'auc': 'roc_auc_ovr'}

cv_results_ada = cross_validate(ada, X_train, y_train, cv=5, scoring=scoring)

cv_df_ada = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1', 'AUC'],
    'Mean': [cv_results_ada['test_acc'].mean(), cv_results_ada['test_recall'].mean(), cv_results_ada['test_f1'].mean(), cv_results_ada['test_auc'].mean()],
    'Std': [cv_results_ada['test_acc'].std(), cv_results_ada['test_recall'].std(), cv_results_ada['test_f1'].std(), cv_results_ada['test_auc'].std()]
})

ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
y_prob_ada = ada.predict_proba(X_test)
test_acc_ada, test_recall_ada, test_f1_ada, test_auc_ada = evaluate_ensemble(y_test, y_pred_ada, y_prob_ada)

test_df_ada = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1', 'AUC'],
    'Value': [test_acc_ada, test_recall_ada, test_f1_ada, test_auc_ada]
})

# XGBoost
xgb = XGBClassifier(eval_metric='mlogloss')

cv_results_xgb = cross_validate(xgb, X_train, y_train_xgb, cv=5, scoring=scoring)

cv_df_xgb = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1', 'AUC'],
    'Mean': [cv_results_xgb['test_acc'].mean(), cv_results_xgb['test_recall'].mean(), cv_results_xgb['test_f1'].mean(), cv_results_xgb['test_auc'].mean()],
    'Std': [cv_results_xgb['test_acc'].std(), cv_results_xgb['test_recall'].std(), cv_results_xgb['test_f1'].std(), cv_results_xgb['test_auc'].std()]
})

xgb.fit(X_train, y_train_xgb)
y_pred_xgb = xgb.predict(X_test) + 1  # Powrót do oryginalnych etykiet
y_prob_xgb = xgb.predict_proba(X_test)
test_acc_xgb, test_recall_xgb, test_f1_xgb, test_auc_xgb = evaluate_ensemble(y_test, y_pred_xgb, y_prob_xgb)

test_df_xgb = pd.DataFrame({
    'Metric': ['ACC', 'Recall', 'F1', 'AUC'],
    'Value': [test_acc_xgb, test_recall_xgb, test_f1_xgb, test_auc_xgb]
})

with pd.ExcelWriter('boosting.xlsx') as writer:
    cv_df_ada.to_excel(writer, sheet_name='ADA CV', index=False)
    test_df_ada.to_excel(writer, sheet_name='ADA Test', index=False)
    cv_df_xgb.to_excel(writer, sheet_name='XGB CV', index=False)
    test_df_xgb.to_excel(writer, sheet_name='XGB Test', index=False)

print("Zadanie 4 ukończone.")

# Zadanie 5 Porównanie
comparison_data = {
    'Method': ['Ensemble Voting', 'Aggregating', 'ADA Boost', 'XG Boost'],
    'ACC': [test_acc, test_acc_agg, test_acc_ada, test_acc_xgb],
    'Recall': [test_recall, test_recall_agg, test_recall_ada, test_recall_xgb],
    'F1': [test_f1, test_f1_agg, test_f1_ada, test_f1_xgb],
    'AUC': [test_auc, test_auc_agg, test_auc_ada, test_auc_xgb]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_excel('comparison.xlsx', index=False)

print("Zadanie 5 ukończone.")