import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def train_nmf(docs, num_topics=10):
    """
    Input: list of tokenized documents, number of topics
    Output: nmf model
    """
    print('Started training NMF model...')
    # create features
    print('Creating features...')
    tfidf_vectorizer = TfidfVectorizer(min_df=3, 
                                       max_df=0.85, 
                                       max_features=5000, 
                                       ngram_range=(1, 2), 
                                       preprocessor=' '.join
                                       )
    tfidf = tfidf_vectorizer.fit_transform(docs)

    print('Training NMF model with tfidf...')
    # build NMF model
    nmf_model = NMF(n_components=num_topics, 
                    init='nndsvd', 
                    random_state=4263)
    
    print('Completed training NMF model!')
    return nmf_model, tfidf, tfidf_vectorizer

## Testing 
# import os

# current_path = os.getcwd()
# df = pd.read_csv(current_path + '/data/clean_reviews.csv', encoding='unicode_escape')

# res, tfidf, tfidf_vectorizer = train_nmf(list(df['stem_clean_text'].apply(lambda x: x.split())), num_topics=10)

# W = res.fit_transform(tfidf)
# H = res.components_

# n_top_words = 10
# feature_names = tfidf_vectorizer.get_feature_names_out()
# for topic_idx, topic in enumerate(res.components_):
#   # print(topic_idx, " ", topic)
#   top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
#   top_features = [feature_names[i] for i in top_features_ind]
#   print(topic_idx, " ", top_features)