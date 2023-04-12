from transformations import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import xgboost


def train_xgboost(X_train_tf, X_train_word, X_test_tf, X_test_word, y_train, y_test, metric= "accuracy"):
    # Hyperparameters for optimization
    params = {
        "learning_rate": [0.001, 0.01, 0.1],
        "max_depth": [3, 5, 7],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "subsample": [0.5, 0.7, 0.9],
        "eval_metric": ["logloss"]
    }

    classifier = xgboost.XGBClassifier()

    # select best model
    grid_search = GridSearchCV(classifier, param_grid=params, n_jobs=-1, scoring= metric, cv=3,verbose=0)
    grid_search.fit(X_train_tf, y_train)
    # Print out the accuracy scores for each model trained
    print("Accuracy scores for each model trained using TF-IDF in GridSearchCV:")
    print(grid_search.cv_results_['mean_test_score'])
    tf_best_estimator = grid_search.best_estimator_
    tf_best_score = grid_search.best_score_

    grid_search.fit(X_train_word, y_train)
    # Print out the accuracy scores for each model trained
    print("Accuracy scores for each model trained using word2vec in GridSearchCV:")
    print(grid_search.cv_results_['mean_test_score'])
    word_best_estimator = grid_search.best_estimator_
    word_best_score = grid_search.best_score_

    # return best model and 
    if tf_best_score < word_best_score:
        print("Training on X_train_word gives the best estimator and score: " + str(word_best_score))
        # Using the final xgboost tuning paramters, refit the model with the entire training set
        word_best_estimator.fit(X_train_word, y_train)
        y_pred = word_best_estimator.predict(X_test_word)
        y_pred_proba = word_best_estimator.predict_proba(X_test_tf)[:, 1]
        return word_best_estimator, y_pred, y_pred_proba
    else:
        print("Training on X_train_tf gives the best estimator and score: " + str(tf_best_score))
        # Using the final xgboost tuning paramters, refit the model with the entire training set
        tf_best_estimator.fit(X_train_tf, y_train)
        y_pred = tf_best_estimator.predict(X_test_tf)
        y_pred_proba = tf_best_estimator.predict_proba(X_test_tf)[:, 1]
        return tf_best_estimator, y_pred, y_pred_proba
    

                       
                       
                       
                       
                       