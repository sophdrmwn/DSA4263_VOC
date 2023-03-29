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

def train_xgboost(df):
    # compare the results of two feature engineering methods
    X_tf,y_tf = tf_idf(df)
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
        X_tf, y_tf, test_size=0.2, stratify=y_tf, random_state=4263
    )
    X_word, y_word = word2vec(df)
    X_train_word, X_test_word, y_train_word, y_test_word = train_test_split(
        X_word, y_word, test_size=0.2, stratify=y_word, random_state=4263
    )

    # Hyperparameters for optimization
    params = {
        "learning_rate": [0.001, 0.01, 0.1, 1],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0, 4, 0.5, 0.7]

    }
    classifier = xgboost.XGBClassifier()

    # select best model
    grid_search_tf = GridSearchCV(classifier, param_grid=params, n_job=-1, scoring="accuracy", cv= 5,verbose=3)
    grid_search_tf.fit(X_train_tf, y_train_tf)
    tf_best_estimator = grid_search_tf.best_estimator_
    tf_best_score = grid_search_tf.best_score

    grid_search_word = GridSearchCV(classifier, param_grid=params, n_job=-1, scoring="accuracy", cv= 5,verbose=3)
    grid_search_word.fit(X_train_word, y_train_word)
    word_best_estimator = grid_search_word.best_estimator_
    word_best_score = grid_search_word.best_score

    # return best model
    if tf_best_score < word_best_score:
        return word_best_estimator
    else:
        return tf_best_estimator



