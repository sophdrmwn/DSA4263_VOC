import pandas as pd 
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

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




