# reference from https://github.com/aishwaryaprabhat/machine_learning_dockerized

from flask import Flask
from flask import request
from flasgger import Swagger

import os
import pickle
import src.transformations as c

# load vectorizer and models
current_path = os.getcwd()
vectorizer = pickle.load(open(current_path+"/models/tfidfvectorizer.pickle", "rb"))
nmf_model = pickle.load(open(current_path+"/models/nmf_model.pickle", "rb"))

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
    # TODO: predict sentiment

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
    res = "The review is a positive/ negative comment about " + topic + "."
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)