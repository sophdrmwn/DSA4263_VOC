import pandas as pd
import numpy as np

import gensim
from gensim import corpora, models

def train_lda(docs, num_topics=10):
    """
    Input: list of tokenized documents, number of topics
    Output: list of 2 lda models (each trained with BoW or tfidf)
    """
    print('Started training LDA models...')
    # create features
    print('Creating features...')
    # create dictionary
    dictionary = corpora.Dictionary(docs)
    # convert docs into BoW format
    corpus_bow = [dictionary.doc2bow(doc) for doc in docs]

    # create tfidf model
    tfidf = models.TfidfModel(corpus_bow)
    # convert docs into tfidf format
    corpus_tfidf = tfidf[corpus_bow]

    print('Training LDA model with BoW...')
    # build LDA model
    bow_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_bow, 
                                                    id2word=dictionary, 
                                                    num_topics=num_topics, 
                                                    random_state=4263, 
                                                    update_every=1, 
                                                    chunksize=100, 
                                                    passes=50, 
                                                    alpha='auto',
                                                    per_word_topics=True)
    print('Training LDA model with tfidf...')
    tfidf_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, 
                                                      id2word=dictionary, 
                                                      num_topics=num_topics, 
                                                      random_state=4263, 
                                                      update_every=1, 
                                                      chunksize=100, 
                                                      passes=50, 
                                                      alpha='auto', 
                                                      per_word_topics=True)
    
    print('Completed training LDA models!')
    return [bow_lda_model, tfidf_lda_model]
