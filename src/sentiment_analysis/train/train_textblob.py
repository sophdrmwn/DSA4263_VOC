import pandas as pd
from textblob import TextBlob

def train_textblob(df):
    
    df['TextBlob_subjectivity'] = df['stem_clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['TextBlob_polarity'] = df['stem_clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['TextBlob_sentiment'] = df['TextBlob_polarity'].apply(textblob_sentiment)

    return df
    
def textblob_sentiment(value):

    if value < 0:
        return 'negative'

    elif value > 0:
        return 'positive'

    else:
        return 'neutral'