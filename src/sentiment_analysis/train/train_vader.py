import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def train_vader(df):
    
    df['Vader_compound'] = df['stem_clean_text'].apply(sentiment_scores)
    df['Vader_sentiment'] = df['Vader_compound'].apply(vader_sentiment)


    return df

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict['compound']

def vader_sentiment(value):

    if value >= 0.05:
        return 'positive'

    elif value <= -0.05:
        return 'negative'
    
    else:
        return 'neutral'

