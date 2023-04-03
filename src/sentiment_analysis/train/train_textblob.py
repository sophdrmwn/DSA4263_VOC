import pandas as pd
from textblob import TextBlob

def train_textblob(df):
    
    df['TextBlob_sentiment'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity).apply(getTextblobSentiment)
    df['TextBlob_sentiment_num'] = df.TextBlob_sentiment.map({"positive": 1, "negative": 0})

    return df

def getTextblobSentiment(value):
    
    if value < 0:
        return 'negative'

    else:
        return 'positive'