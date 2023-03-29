import src.transformations*
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost

from sklearn.model_selection import train_test_split
X,y = tf_idf(df_clean)
X, y = word2vec(df_clean)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=4263
    )

def train_xgboost(X,y):
    # choose one of feature engineering methods: tf_idf or word2vec
    X_tfidf,y = tf_idf(df_clean)
    X_word2vec,y = word2vec(df_clean)
    # split train-test data
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, stratify=y, random_state=4263
    )
    X_word2vec, X_test_word2vec, y_train, y_test = train_test_split(
        X_word2vec, y, test_size = 0.2, stratify = y, random_state=4263
    )

    # Hyperparameters for optimization
    params = {
        "learning_rate": [0.001, 0.01, 0.1, 1],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0, 4, 0.5, 0.7]

    }
    classifier = xgboost.XGBClassifier()
    random_search_tfidf = RandomizedSearchCV(classifier, param_distributions=params, n_iter=10, scoring="accuracy", verbose=3)
    random_search.fit(X, y)
    return random_search.best_estimator_
##