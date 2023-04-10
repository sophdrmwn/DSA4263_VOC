import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
    
def eval_textblob(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    return {"accuracy":accuracy, "recall":recall, "precision":precision,"f1":f1, "auc":auc}