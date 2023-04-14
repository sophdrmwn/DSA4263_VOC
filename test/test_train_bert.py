from src.sentiment_analysis.train.train_bert import compute_metrics, pred_bert

import numpy as np
from unittest.mock import patch
from numpy.testing import assert_array_equal

def test_compute_metrics():
    # Mock prediction and labels
    pred = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7]])
    labels = np.array([0, 1, 1])

    # Call the function being tested
    result = compute_metrics((pred, labels))

    assert type(result) == dict
    assert (result['accuracy'] > 0 and result['accuracy'] <= 1)

def test_pred_bert():

    text_list = ["This movie is great!", "This movie is terrible!"]
    expected_list = [1, 0]
    y_pred, y_score = pred_bert(text_list, 'models/bert-full-train', True)

    assert_array_equal(np.array(y_pred), np.array(expected_list))
    assert y_score[0] > 0.5
    assert y_score[1] < 0.5

