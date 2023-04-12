import pandas as pd
import numpy as np

import gensim
from gensim import corpora, models

def train_lda(df, num_topics=20, n_top_words=10, text_col='stem_clean_text'):
    """
    Trains a LDA model with the data and parameters provided.

    Args:
        df (dataframe): Dataframe with a column of text reviews
        num_topics (int, optional): Expected total number of topics mentioned in the reviews
        n_top_words (int, optional): Number of top words of each topic to return
        text_col (str, optional): Name of column containing the reviews; default set as 'stem_clean_text'
    
    Returns:
        A tuple containing:
            - topic_words (list): A list of tuples, each containing the topic number and its corresponding n top words as strings.
            - pred (list): A list of integers, each representing the predicted topic number for the corresponding data point.
            - lda_model (gensim.models.ldamodel.LdaModel): The trained LDA model.
    """
    # create features
    docs = list(df[text_col].apply(lambda x: x.split()))
    # create dictionary
    dictionary = corpora.Dictionary(docs)
    # convert docs into BoW format
    corpus_bow = [dictionary.doc2bow(doc) for doc in docs]
    
    # build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_bow, 
                                                id2word=dictionary, 
                                                num_topics=num_topics, 
                                                random_state=4263, 
                                                update_every=1, 
                                                chunksize=100, 
                                                passes=50, 
                                                per_word_topics=True)

    # get list of predicted topics
    pred = []
    for i, row in enumerate(lda_model[corpus_bow]):
        row = row[0]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        topic = row[0][0]
        pred.append(topic)
    
    # get n_top_words of each topic
    topic_words = lda_model.print_topics(num_words=n_top_words)
    
    return topic_words, pred, lda_model