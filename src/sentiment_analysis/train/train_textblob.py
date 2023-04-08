import pandas as pd
from textblob import TextBlob

def train_textblob(df):
    
    return df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity).apply(getTextblobSentiment)


def getTextblobSentiment(value):
    
    if value < 0:
        return 'negative'

    else:
        return 'positive'