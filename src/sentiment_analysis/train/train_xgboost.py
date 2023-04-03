from src.transformations import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost

# loading raw data
current_path = os.getcwd()
df = pd.read_csv(current_path + '/data/reviews.csv', encoding='unicode_escape')

# delete after including this part in pipeline
# data cleaning
df['clean_text'] = df['Text'].apply(lambda x: get_cleantext(x))
df['Sentiment_num'] = df.Sentiment.map({"positive": 1, "negative": 0})
X = df['clean_text'].to_list()
y = df['Sentiment_num'].to_list()
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=4263
    )

def train_xgboost(X_train, y_train):
    # compare the results of two feature engineering methods
    X_train_tf = tf_idf(X_train)
    X_train_word =word2vec(X_train)

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
    print("Start training xgboost model with feature engineering TF-IDF.\n")
    grid_search_tf = GridSearchCV(classifier, param_grid=params, n_job=-1, scoring="accuracy", cv= 5,verbose=3)
    grid_search_tf.fit(X_train_tf, y_train)
    tf_best_estimator = grid_search_tf.best_estimator_
    tf_best_score = grid_search_tf.best_score

    print("Start training xgboost model with feature engineering word2vec.\n")
    grid_search_word = GridSearchCV(classifier, param_grid=params, n_job=-1, scoring="accuracy", cv= 5,verbose=3)
    grid_search_word.fit(X_train_word, y_train)
    word_best_estimator = grid_search_word.best_estimator_
    word_best_score = grid_search_word.best_score

    # return best model
    if tf_best_score < word_best_score:
        print("Successfully run the model with word2vec and return best estimator and score\n")
        return word_best_estimator, word_best_score
    else:
        print("Successfully run the model with word2vec and return best estimator and score\n")
        return tf_best_estimator, tf_best_score
