from src.transformations import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost

# loadind raw data
current_path = os.getcwd()
df = pd.read_csv(current_path + '/data/reviews.csv', encoding='unicode_escape')

# data cleaning
df['clean_text'] = df['Text'].apply(lambda x: get_cleantext(x))
df['stem_clean_text'] = df['Text'].apply(lambda x: get_cleantext(x, stemming=True))

# choose one of feature engineering methods & train-test split
# TF_IDF or word2vec, commit one of the lines
X,y = tf_idf(df)
# word2vec
#X, y = word2vec(df)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=4263
    )

def train_xgboost(X_train,y_train):
    # Hyperparameters for optimization
    params = {
        "learning_rate": [0.001, 0.01, 0.1, 1],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0, 4, 0.5, 0.7]

    }
    classifier = xgboost.XGBClassifier()
    grid_search = GridSearchCV(classifier, param_grid=params, n_job=-1, scoring="accuracy", cv= 5,verbose=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
