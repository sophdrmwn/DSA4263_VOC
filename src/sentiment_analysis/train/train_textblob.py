import pandas as pd
from textblob import TextBlob

def train_textblob(data):
    
    TextBlob_sentiment = []

    for item in data:
        TextBlob_sentiment.append(getTextblobSentiment(TextBlob(item).sentiment.polarity))

    return TextBlob_sentiment


def getTextblobSentiment(value):
    
    if value < 0:
        return 0

    else:
        return 1