import pandas as pd
import sys
sys.path.insert(0, '../src')

from topic_modelling.train.train_lda import *

test_df = pd.DataFrame(['i liked the orange juice it was not too sweet or sour', 
                        'i did not like the coffee to tasted too bitter', 
                        'dog food was not good it smelled so stale even though the expiration date is 2 years away', 
                        'cat food smelled better than the dog food',
                        'orange juice was too sweet'], columns=['stem_clean_text'])

def test_train_func_returns_threeobjs():
    # check that train function returns 3 objects
    assert len(train_lda(test_df, num_topics=3))==3

def test_train_func_returns_topicwords():
    # check that train function returns topic words as a list with length num_topics
    num_topics=3
    topic_words, pred, lda_model = train_lda(test_df, num_topics=num_topics)
    assert (len(topic_words)==num_topics) & (type(topic_words)==list)

def test_train_func_returns_pred():
    # check that train function returns predicted topics as a list which is same length as the number of reviews
    num_topics=3
    topic_words, pred, lda_model = train_lda(test_df, num_topics=num_topics)
    assert (type(pred)==list) & (len(pred)==len(test_df))