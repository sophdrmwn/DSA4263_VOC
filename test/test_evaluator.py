from src.sentiment_analysis.train.evaluator import *
import numpy as np

def test_get_metrics():
    y_pred = [0, 1, 1, 0, 1, 0]
    y_test = [1, 1, 0, 0, 1, 1]
    results_dict = {}
    model_name = "test_model"
    
    expected_results = {"test_model": {"accuracy": 0.5, 
                                       "recall": 0.5, 
                                       "precision": 0.5, 
                                       "f1": 0.5, 
                                       "auc": 0.5}}
    
    assert get_metrics(y_pred, y_test, results_dict, model_name) == expected_results