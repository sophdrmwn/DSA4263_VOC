import pandas as pd
import numpy as np

from src.transformations import skl_tfidf
from sklearn.decomposition import NMF

def train_nmf(df, num_topics=20, n_top_words=10):
    """
    Input: df with 'stem_clean_text' column, number of topics, top n number of words for each topic
    Output: top words in each topic, list of predicted topics, nmf model
    """
    # create tfidf
    tfidf, tfidf_vectorizer = skl_tfidf(df)

    # build NMF model
    nmf_model = NMF(n_components=num_topics, 
                    init='nndsvd', 
                    random_state=4263)
    nmf_model.fit(tfidf)

    # get list of predicted topics
    pred = list(pd.DataFrame(nmf_model.transform(tfidf)).idxmax(axis=1))

    # get n_top_words of each topic
    topic_words = []
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf_model.components_):
       top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
       top_features = [feature_names[i] for i in top_features_ind]
       topic_words.append([topic_idx, top_features])
  
    return topic_words, pred, nmf_model

## Testing 
# import os

# current_path = os.getcwd()
# df = pd.read_csv(current_path + '/data/clean_reviews.csv', encoding='unicode_escape')

# topic_words, pred, nmf_model = train_nmf(df)