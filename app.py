# reference from https://github.com/aishwaryaprabhat/machine_learning_dockerized

from flask import Flask
from flask import request
from flasgger import Swagger
from transformers import pipeline, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

import os
import pickle
import src.transformations as c

# load vectorizer and models
current_path = os.getcwd()
vectorizer = pickle.load(open(current_path+"/models/tfidfvectorizer.pickle", "rb"))
nmf_model = pickle.load(open(current_path+"/models/nmf_model.pickle", "rb"))

sentiment_analysis = pipeline(
  'sentiment-analysis', 
  model = current_path + '/models/bert-full-train', 
  tokenizer = 'bert-base-uncased', 
  truncation = True, 
  max_length = 512, 
  padding = True
)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')

def predict():
    """Example endpoint returning a prediction of sentiment and topic
    ---
    parameters:
      - name: review
        in: query
        type: string
        required: true
    responses:
      200:
        description: Successfully predicted sentiment and topic!

    """
    review = request.args.get('review')
    
    # predict sentiment
    result = sentiment_analysis(review)
    if int(result[0]["label"][-1:]) == 1:
      sentiment_prob = float(result[0]["score"])
    else:
      sentiment_prob = 1 - float(result[0]["score"])
    if sentiment_prob>0.5:
       sentiment = "positive"
    else:
       sentiment = "negative"
       


    # predict topic
    clean_review = c.get_cleantext(review, stemming=True)
    tfidf_review = vectorizer.transform([clean_review])
    topic_num = nmf_model.transform(tfidf_review).argmax()
    topic_labels = {0: 'Taste', 1: 'Coffee', 2: 'Tea', 3: 'Amazon', 4: 'Taste/ Price', 5: 'Healthy Snacks', 
                        6: 'Dog Food', 7: 'Healthy Carbohydrates', 8: 'Cold Beverages', 9: 'Hot Beverages', 
                        10: 'Flavour', 11: 'Chips', 12: 'Delivery Feedback', 13: 'Taste', 14: 'Saltiness', 
                        15: 'Recommendations', 16: 'Trying', 17: 'Alternative Sources', 18: 'Cat/ Baby Food', 
                        19: 'Cooking Ingredients'}
    topic = topic_labels[topic_num]

    # prepare result
    review = 'This is a {0:s} review about {1:s} with a sentiment probability of {2:.5f}.'

    return review.format(sentiment, topic, sentiment_prob)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)