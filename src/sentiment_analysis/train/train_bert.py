from sklearn.metrics import accuracy_score
from transformers import pipeline, BertForSequenceClassification, BertTokenizer, Trainer
from ray import tune
import os
import pandas as pd
import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def compute_metrics(p):

    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {'accuracy': accuracy}

def ray_hp_space(trial):

    return {
        'learning_rate': tune.grid_search([3e-5, 5e-5]),
        'num_train_epochs': tune.grid_search([2, 4])
    }

def model_init(trial):

    return BertForSequenceClassification.from_pretrained('bert-base-uncased')

def tune_bert(X_train, X_test, y_train, y_test):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train_tokenized = tokenizer(X_train, padding = True, truncation = True, max_length = 512)
    X_test_tokenized = tokenizer(X_test, padding = True, truncation = True, max_length = 512)

    train_dataset = Dataset(X_train_tokenized, y_train)
    test_dataset = Dataset(X_test_tokenized, y_test)

    trainer = Trainer(
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        compute_metrics = compute_metrics,
        model_init = model_init
    )

    best_trial = trainer.hyperparameter_search(
        direction = "maximize",
        backend = "ray",
        hp_space = ray_hp_space,
        n_trials = 1
    )

    return best_trial

def pred_bert_new(filename = 'reviews_test.csv', col_name = 'Text', model_name = 'bert-full-train'):

    current_path = os.getcwd()
    root_path = os.path.dirname(current_path)
    df = pd.read_csv(root_path + '/data/' + filename, encoding='unicode_escape')

    y_pred, y_score = pred_bert(df[col_name].to_list(), return_score = True, model_name = 'bert-full-train')

    df['predicted_sentiment_probability'] = y_score
    df['predicted_sentiment'] = y_pred

    df.to_csv(root_path + '/data/reviews_test_prediction_Group_9.csv')