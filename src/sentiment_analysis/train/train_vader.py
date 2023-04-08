import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

def train_vader(df):
    
    df['Vader_polarity'] = df['Text'].apply(getPolarity)
    scaler = MinMaxScaler()
    df[['Vader_polarity_scaled']] = scaler.fit_transform(df[['Vader_polarity']])
    return df['Vader_polarity_scaled'].apply(getVaderSentiment)
    

def getPolarity(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict['compound']

def getVaderSentiment(value):
    
    if value < 0.5:
        return 'negative'

    else:
        return 'positive'



