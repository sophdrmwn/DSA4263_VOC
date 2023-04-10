from sentiment_analysis.train.train_xgboost import *
from transformations import *
import numpy as np
import xgboost

import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pytest
from gensim.models import KeyedVectors


def test_get_mean_vector():
    wv_dict = {'hello': [1,2,3,4,5], 'world': [5,4,3,2,1]}
    wv = KeyedVectors(vector_size=5)
    wv.add_vectors(list(wv_dict.keys()), list(wv_dict.values()))
    
    text = ['hello', 'world']
    vector = get_mean_vector(text, wv)
    expected_vector = np.array([3., 3., 3., 3., 3.])
    assert np.allclose(vector, expected_vector)

def test_word2vec():
    wv_dict = {'hello': [1,2,3,4,5], 'world': [5,4,3,2,1]}
    wv = KeyedVectors(vector_size=5)
    wv.add_vectors(list(wv_dict.keys()), list(wv_dict.values()))
    
    X = ['hello world', 'this is a test']
    result = word2vec(X, wv=wv)
    expected_result = np.array([[3., 3., 3., 3., 3.], [0., 0., 0., 0., 0.]])
    
    assert np.allclose(result, expected_result)

def test_tf_idf():
    # Create some test data
    X = ['This is a test', 'Another test string', 'Yet another test string']
    # Call the function to obtain the feature matrix and the vectorizer
    X_transformed, vectorizer = tf_idf(X)
    # Check that the feature matrix has the expected dimensions
    assert X_transformed.shape == (3, 4)

def test_train_xgboost():
    X_train_tf = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_train_word = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1],[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    y_train = np.array([0, 1, 0, 0, 1, 0])
    best_estimator, fe_method = train_xgboost(X_train_tf, X_train_word, y_train, metric="accuracy")
    assert fe_method in ["tfidf", "word"]
    assert isinstance(best_estimator, xgboost.XGBClassifier)

def test_eval_xgboost():
    X_train_tf = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_train_word = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1],[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    X_test_tf = np.array([[3, 2, 1], [6, 5, 4]])
    X_test_word = np.array([[7, 8, 9], [1, 2, 3]])
    y_train = np.array([0, 1, 0, 0, 1, 0])
    y_test = np.array([1, 0])
    best_estimator, fe_method = train_xgboost(X_train_tf, X_train_word, y_train, metric="accuracy")
    results = eval_xgboost(best_estimator, fe_method, X_train_tf, X_train_word, X_test_tf, X_test_word, y_train, y_test)
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "recall" in results
    assert "precision" in results
    assert "f1" in results
    assert "auc" in results
