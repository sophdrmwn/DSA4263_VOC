import pandas as pd
from textblob import TextBlob
import numpy as np
import scipy
from scipy import stats

def train_textblob(data):
    
    TextBlob_sentiment = []

    for item in data:
        # offset polarity score (in float) from range [-1,1] to [0,1]
        offset = (TextBlob(item).sentiment.polarity + 1)/2
        # convert offset float score in [0,1] to integer value in range [1,20]
        binned = np.digitize(20 * offset, np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])) + 1
        # simulate probabilities of each class based on a normal distribution
        simulated_probs = scipy.stats.norm.pdf(np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]), binned, scale=0.5)
        # add index of highest probability to list
        TextBlob_sentiment.append(simulated_probs.argmax())

    final =  [0 if x <= 10  else 1 for x in TextBlob_sentiment]

    return final
