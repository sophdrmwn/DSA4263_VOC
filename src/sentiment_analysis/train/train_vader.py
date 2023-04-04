import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def train_vader(df):
    
    df['Vader_polarity'] = df['Text'].apply(getPolarity)
    scaler = MinMaxScaler()
    df[['Vader_polarity_scaled']] = scaler.fit_transform(df[['Vader_polarity']])
    df['Vader_sentiment'] = df['Vader_polarity_scaled'].apply(getVaderSentiment)
    
    y_pred = df['Vader_sentiment']
    y_true = df['Sentiment']

    return df[['Sentiment','Vader_sentiment']], accuracy_score(y_true, y_pred)


def getPolarity(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict['compound']

def getVaderSentiment(value):
    
    if value < 0.5:
        return 'negative'

    else:
        return 'positive'



