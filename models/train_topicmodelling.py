import os
import pandas as pd
import pickle

import src.transformations as c
import src.topic_modelling.train.train_nmf as tm

current_path = os.getcwd()
root_path = os.path.dirname(current_path)

def train_save_topic_model(df, num_topics=20, n_top_words=10):
    """
    Input: df with 'stem_clean_text' column, number of topics, top n number of words for each topic
    This function trains and saves a NMF model (and tfidf vectorizer) with the data and parameters provided.
    Pickles are saved under the models folder.
    """
    nmf_topic_words, nmf_pred, nmf_model = tm.train_nmf(df, 
                                                        num_topics=num_topics, 
                                                        n_top_words=n_top_words, 
                                                        save=True)
    
def predict_topic(review):
    """
    Input: 1 review (String)
    Output: predicted topic for that review using the saved model
    """
    vectorizer = pickle.load(open(root_path+"/models/tfidfvectorizer.pickle", "rb"))
    nmf_model = pickle.load(open(root_path+"/models/nmf_model.pickle", "rb"))

    clean_review = c.get_cleantext(review, stemming=True)
    tfidf_review = vectorizer.transform([clean_review])
    topic_num = nmf_model.transform(tfidf_review).argmax()

    # label topics
    topic_labels = {0: 'Taste', 1: 'Coffee', 2: 'Tea', 3: 'Amazon', 4: 'Taste/ Price', 5: 'Healthy Snacks', 
                    6: 'Dog Food', 7: 'Healthy Carbohydrates', 8: 'Cold Beverages', 9: 'Hot Beverages', 
                    10: 'Flavour', 11: 'Chips', 12: 'Delivery Feedback', 13: 'Taste', 14: 'Saltiness', 
                    15: 'Recommendations', 16: 'Trying', 17: 'Alternative Sources', 18: 'Cat/ Baby Food', 
                    19: 'Cooking Ingredients'}
    
    return topic_labels[topic_num]