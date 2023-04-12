from src.sentiment_analysis.train.evaluator import *
import numpy as np
import unittest


def test_get_metrics():
    y_pred = [1, 0, 1, 1, 0, 0, 1]
    y_test = [1, 1, 1, 0, 0, 1, 0]
    results_dict = {}
    model_name = "Test Model"
    y_pred_score = np.array([0.8, 0.6, 0.9, 0.7, 0.2, 0.4, 0.1])

    expected_acc = accuracy_score(y_test, y_pred)
    expected_pre = precision_score(y_test, y_pred)
    expected_recall = recall_score(y_test, y_pred)
    expected_f1 = f1_score(y_test, y_pred)
    expected_auc = roc_auc_score(y_test, y_pred_score)

    results_dict = get_metrics(y_pred, y_test, results_dict, model_name, y_pred_score)

    assert results_dict[model_name]["accuracy"] == expected_acc
    assert results_dict[model_name]["precision"] == expected_pre
    assert results_dict[model_name]["recall"] == expected_recall
    assert results_dict[model_name]["f1"] == expected_f1
    assert results_dict[model_name]["auc"] == expected_auc

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)