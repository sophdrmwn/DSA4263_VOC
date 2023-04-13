from src.sentiment_analysis.train.train_xgboost import *
import numpy as np
import xgboost
import unittest

def test_train_xgboost():
    X_train_tf = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_train_word = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1],[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    y_train = [0, 1, 0, 0, 1, 0]
    X_test_tf = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_test_word = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    y_test = [0, 1, 0]
    best_estimator, y_pred, y_pred_proba = train_xgboost(X_train_tf, X_train_word, X_test_tf, X_test_word, y_train, y_test, metric= "accuracy")
    # check if the model is of the correct type
    assert isinstance(best_estimator, xgboost.XGBClassifier)
    # Check that y_pred is binary
    print(y_pred)
    assert set(y_pred).issubset(set([0, 1]))
    # Check that y_pred_proba is within reasonable range
    print(y_pred_proba)
    assert all(p >= 0 and p <= 1 for p in y_pred_proba)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)




