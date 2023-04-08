import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score

def train_textblob(df):
    
    df['TextBlob_sentiment'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity).apply(getTextblobSentiment)

    return df


def getTextblobSentiment(value):
    
    if value < 0:
        return 'negative'

    else:
        return 'positive'