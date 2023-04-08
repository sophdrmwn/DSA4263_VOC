import pandas as pd
import numpy as np

import gensim
from gensim import corpora, models

def train_lda(df, num_topics=20, n_top_words=10, text_col='stem_clean_text'):
    """
    Input: df, number of topics, top n number of words for each topic, text column name
    Output: top words in each topic, list of predicted topics, lda model
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