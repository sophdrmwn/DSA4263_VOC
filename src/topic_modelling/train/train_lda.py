import pandas as pd
import numpy as np

import gensim
from gensim import corpora, models

def train_lda(df, num_topics=20, fe='bow'):
    """
    Input: df with 'stem_clean_text' column, number of topics, feature engineering method 'tfidf'/'bow'
    Output: list of predicted topics, lda model
    """
    # create features
    docs = list(df['stem_clean_text'].apply(lambda x: x.split()))
    # create dictionary
    dictionary = corpora.Dictionary(docs)
    # convert docs into BoW format
    corpus_bow = [dictionary.doc2bow(doc) for doc in docs]
    corpus = corpus_bow

    if fe=='tfidf':
        # create tfidf model
        tfidf = models.TfidfModel(corpus_bow)
        # convert docs into tfidf format
        corpus_tfidf = tfidf[corpus_bow]
        corpus = corpus_tfidf
    
    # build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                                id2word=dictionary, 
                                                num_topics=num_topics, 
                                                random_state=4263, 
                                                update_every=1, 
                                                chunksize=100, 
                                                passes=50, 
                                                alpha='auto',
                                                per_word_topics=True)

    pred = []
    for i, row in enumerate(lda_model[corpus]):
        row = row[0]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        topic = row[0][0]
        pred.append(topic)
    
    return pred, lda_model
