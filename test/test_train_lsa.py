import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../src')

from topic_modelling.train.train_lsa import *

test_df = pd.DataFrame({'away': [0, 0, 1, 0, 0],
                'bitter': [0, 1, 0, 0, 0],
                'cat': [0, 0, 0, 1, 0],
                'coffee': [0, 1, 0, 0, 0],
                'date': [0, 0, 1, 0, 0],
                'dog': [0, 0, 1, 1, 0],
                'expiration': [0, 0, 1, 0, 0],           
                'food': [0, 0, 1, 2, 0],
                'good': [0, 0, 1, 0, 0],
                'juice': [1, 0, 0, 0, 1],
                'orange': [1, 0, 0, 0, 1],
                'smell': [0, 0, 1, 1, 0],
                'sour': [1, 0, 0, 0, 0],
                'stale': [0, 0, 1, 0, 0],
                'sweet': [1, 0, 0, 0, 1],
                'taste': [0, 1, 0, 0, 0],
                'though': [0, 0, 1, 0, 0],
                'well': [0, 0, 0, 1, 0],
                'year': [0, 0, 1, 0, 0]})

def test_train_func_returns_twobjs():
    # check that train function returns 2 objects
    assert len(lsa(test_df, n=3))==2

def test_train_func_returns_topicwords():
    # check that train function returns topic words as a df
    num_topics=3
    topics, pred = lsa(test_df, n=num_topics)
    assert (topics.shape==(num_topics,10))

def test_train_func_returns_pred():
    # check that train function returns predicted topics as an array which is same length as the number of reviews
    num_topics=3
    topics, pred = lsa(test_df, n=num_topics)
    assert (type(pred)==np.ndarray) & (len(pred)==len(test_df))