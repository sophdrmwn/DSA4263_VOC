from src.sentiment_analysis.train.train_bert import *
from models.train_sentimentanalysis import *

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
    assert result.keys() == ['accuracy']
    assert (result['accuracy'] < 0 and result['accuracy'] < 1)

def test_pred_bert():
    text_single = "This movie is great!"
    expected_single = "Positive review"

    text_list = ["This movie is great!", "This movie is terrible!"]
    expected_list = [1, 0]

    with patch('transformers.pipeline') as mock_pipeline:
        # Test single text input
        mock_pipeline.return_value = [{"label": "LABEL_1"}]
        output = pred_bert(text_single)
        assert output == expected_single

        # Test list input, return_score = False
        mock_pipeline.return_value = [{"label": "LABEL_1"}, {"label": "LABEL_0"}]
        output = pred_bert(text_list)
        assert_array_equal(np.array(output), np.array(expected_list))

        # Test list input, return_score = True
        mock_pipeline.return_value = [{"label": "LABEL_1", "score": 0.9}, {"label": "LABEL_0", "score": 0.1}]
        output, score = pred_bert(text_list, return_score=True)
        assert_array_equal(np.array(output), np.array(expected_list))
        assert_array_equal(np.array(score), np.array([0.9, 0.1]))