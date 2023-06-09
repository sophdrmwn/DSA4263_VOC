from sklearn.metrics import accuracy_score
from transformers import pipeline, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from ray import tune
import os
import pandas as pd
import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading text data for BERT model.

    Parameters:
        encodings (dict): Dictionary containing input_ids and attention_mask
                          for the encoded text data.
        labels (list): List of labels corresponding to the text data. Default: None

    Methods:
        __getitem__(idx):
            Returns a dictionary containing 'input_ids', 'attention_mask',
            and 'labels' (if available) at the specified index.

        __len__():
            Returns the length of the input_ids tensor.
    """
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
    """
    Computes and returns the accuracy score for predictions.

    Parameters:
        p (tuple): Tuple containing predicted probabilities and labels.

    Returns:
        dict: Dictionary containing the accuracy score.
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {'accuracy': accuracy}

def ray_hp_space(trial):
    """
    Returns a dictionary containing chosen hyperparameters for hyperparameter search.

    Parameters:
        trial: Object representing a single trial of a hyperparameter search.

    Returns:
        dict: Dictionary containing chosen hyperparameters to be tuned for a BERT model.
    """
    return {
        'learning_rate': tune.grid_search([3e-5, 5e-5]),
        'num_train_epochs': tune.grid_search([2, 4])
    }

def model_init(trial):
    """
    Initializes and returns a BERT model.

    Parameters:
        trial: Object representing a single trial of a hyperparameter search.

    Returns:
        BertForSequenceClassification: BERT model for sequence classification.
    """
    return BertForSequenceClassification.from_pretrained('bert-base-uncased')

def tune_bert(X_train, X_test, y_train, y_test, use_mps = True):
    """
    Performs hyperparameter tuning for a BERT model.

    Parameters:
        X_train (list): List of text data for training.
        X_test (list): List of text data for testing.
        y_train (list): List of labels for training data.
        y_test (list): List of labels for testing data.
        use_mps (bool): Set as True if running on Mac Silicon machines.

    Returns:
        Object: Object representing the best trial from hyperparameter search.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train_tokenized = tokenizer(X_train, padding = True, truncation = True, max_length = 512)
    X_test_tokenized = tokenizer(X_test, padding = True, truncation = True, max_length = 512)

    train_dataset = Dataset(X_train_tokenized, y_train)
    test_dataset = Dataset(X_test_tokenized, y_test)

    if use_mps:

        training_args = TrainingArguments(
                output_dir = 'results',
                use_mps_device = True
            )
        trainer = Trainer(
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = test_dataset,
            compute_metrics = compute_metrics,
            model_init = model_init
        )
    
    else:

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

def pred_bert(text, model_path = 'bert-full-train', return_score = False, use_mps = True):
    """
    Predict the sentiment of given text(s) using a trained BERT-based sentiment analysis model.

    Parameters:
        text (list): A string or a list of strings.
        model_path (str): The path of the model for prediction to be saved at.
        return_score (bool): If True, the function returns both the predicted label and the score.
        use_mps (bool): If True, the model will use the MPS device for faster training.

    Returns:
        If return_score is True:
            A tuple of two lists, where the first list contains the predicted labels and the second list contains the scores.

        If return_score is False:
            A list of predicted labels.
    """

    if use_mps:
        sentiment_analysis = pipeline(
            'sentiment-analysis', 
            model = model_path, 
            tokenizer = 'bert-base-uncased', 
            truncation = True, 
            max_length = 512, 
            padding = True, 
            device = 'mps'
        )
    else:
        sentiment_analysis = pipeline(
            'sentiment-analysis', 
            model = model_path, 
            tokenizer = 'bert-base-uncased', 
            truncation = True, 
            max_length = 512, 
            padding = True
        )
    
    y_pred = []
    y_score = []
    for review in text:
        result = sentiment_analysis(review)
        y_pred.append(int(result[0]["label"][-1:]))
        if int(result[0]["label"][-1:]) == 1:
            y_score.append(float(result[0]["score"]))
        else:
            y_score.append(1 - float(result[0]["score"]))

    if return_score:
        return y_pred, y_score
        
    else:
        return y_pred

def pred_bert_new(filename = 'reviews_test.csv', col_name = 'Text', model_path = 'bert-full-train', use_mps = True):
    """
    Read a CSV file containing text reviews, predict their sentiments using a trained BERT-based sentiment analysis model,
    and save the predictions to a new CSV file.

    Parameters:
        filename (str): The name of the CSV file to be read.
        col_name (str): The name of the column in the CSV file containing the texts to be predicted.
        model_path (str): The path of the model to be saved at.
        use_mps (bool): If True, the model will use the MPS device for faster training.
    
    Returns:
        DataFrame: Dataframe with appended predicted sentiment probabilities and labels.
    """
    current_path = os.getcwd()
    root_path = os.path.dirname(current_path)
    df = pd.read_csv(root_path + '/data/' + filename, encoding='unicode_escape')

    y_pred, y_score = pred_bert(df[col_name].to_list(), return_score = True, model_name = model_path, use_mps = True)

    df['predicted_sentiment_probability'] = y_score
    df['predicted_sentiment'] = y_pred

    df.to_csv(root_path + '/data/reviews_test_prediction_Group_9.csv')

    return df