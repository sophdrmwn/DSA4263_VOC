import pandas as pd 
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy
from scipy import stats

def train_vader(data):
    Vader_sentiment = []

    for item in data:
        # offset polarity score (in float) from range [-1,1] to [0,1]
        offset = (getPolarity(item) + 1)/2
        # convert offset float score in [0,1] to integer value in range [1,20]
        binned = np.digitize(20 * offset, np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])) + 1
        # simulate probabilities of each class based on a normal distribution
        simulated_probs = scipy.stats.norm.pdf(np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]), binned, scale=0.5)
        # add index of highest probability to list
        Vader_sentiment.append(simulated_probs.argmax())

    final =  [0 if x <= 10  else 1 for x in Vader_sentiment]


    return final
    

def getPolarity(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    return sentiment_dict['compound']





