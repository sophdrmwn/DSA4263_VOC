import pandas as pd
from textblob import TextBlob
import numpy as np
import scipy
from scipy import stats

def train_textblob(data):
    
    TextBlob_sentiment = []

    for item in data:
        offset = (TextBlob(item).sentiment.polarity + 1)/2
        binned = np.digitize(20 * offset, np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])) + 1
        simulated_probs = scipy.stats.norm.pdf(np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]), binned, scale=0.5)
        TextBlob_sentiment.append(simulated_probs.argmax())

    final =  [0 if x <= 10  else 1 for x in TextBlob_sentiment]

    return final
