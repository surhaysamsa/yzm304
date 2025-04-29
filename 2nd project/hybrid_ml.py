import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_svm(train_features, train_labels, test_features, test_labels):
    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)
    acc = accuracy_score(test_labels, preds)
    print(f'SVM Test Accuracy: {acc*100:.2f}%')
    return acc

def run_rf(train_features, train_labels, test_features, test_labels):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)
    acc = accuracy_score(test_labels, preds)
    print(f'Random Forest Test Accuracy: {acc*100:.2f}%')
    return acc
