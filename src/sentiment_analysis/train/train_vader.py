import pandas as pd 
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_vader(data):
    Vader_sentiment = []

    for item in data:
        Vader_sentiment.append(getPolarity(item))
        
    scaler = MinMaxScaler() 
    Vader_sentiment = np.array(Vader_sentiment).reshape(-1,1)
    normalized = scaler.fit_transform(Vader_sentiment)  
    final =  [0 if x < 0.5 else 1 for x in normalized]

    return final
    

def getPolarity(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict['compound']

def eval_vader(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    return {"accuracy":accuracy, "recall":recall, "precision":precision,"f1":f1, "auc":auc}




