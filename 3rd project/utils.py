import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(data):
    texts = list(data.keys())
    labels = np.array([int(v) for v in data.values()])
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts).toarray()
    return X, labels, vectorizer

def preprocess_test_data(data, vectorizer):
    texts = list(data.keys())
    labels = np.array([int(v) for v in data.values()])
    X = vectorizer.transform(texts).toarray()
    return X, labels
