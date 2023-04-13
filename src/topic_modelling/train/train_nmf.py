import os
import pandas as pd
import numpy as np
import pickle

from src.transformations import new_tfidf
from sklearn.decomposition import NMF

current_path = os.getcwd()
root_path = os.path.dirname(current_path)

def train_nmf(df, num_topics=20, n_top_words=10, save=False, text_col='stem_clean_text'):
    """
    Trains a NMF model with the data and parameters provided.

    Args:
        df (dataframe): Dataframe with a column of text reviews
        num_topics (int, optional): Expected total number of topics mentioned in the reviews
        n_top_words (int, optional): Number of top words of each topic to return
        save (Boolean, optional): Option to save the NMF model as pickle file.
        text_col (str, optional): Name of column containing the reviews; default set as 'stem_clean_text'
    
    Returns:
        A tuple containing:
            - topic_words (list): A list of tuples, each containing the topic number and its corresponding n top words as strings.
            - pred (list): A list of integers, each representing the predicted topic number for the corresponding data point.
            - nmf_model (sklearn.decomposition.nmf.NMF): The trained NMF model.
    """
    # create tfidf
    tfidf_df = new_tfidf(df[text_col].tolist(), save=save, ngram_range=(1, 2), max_df=0.85, min_df=3, max_features=5000)
    tfidf = tfidf_df.to_numpy()

    # build NMF model
    nmf_model = NMF(n_components=num_topics, 
                    init='nndsvd', 
                    random_state=4263)
    nmf_model.fit(tfidf)

    # save model if needed
    if save:
        pickle.dump(nmf_model, open(root_path+"/models/nmf_model.pkl", "wb"))

    # get list of predicted topics
    pred = list(pd.DataFrame(nmf_model.transform(tfidf)).idxmax(axis=1))

    # get n_top_words of each topic
    topic_words = []
    feature_names = tfidf_df.columns
    for topic_idx, topic in enumerate(nmf_model.components_):
       top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
       top_features = [feature_names[i] for i in top_features_ind]
       topic_words.append((topic_idx, top_features))
  
    return topic_words, pred, nmf_model