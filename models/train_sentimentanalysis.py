from transformers import pipeline, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
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

def train_bert(X, y, model_name = 'bert-full-train', use_mps = True):
    """
    Train a BERT-based sentiment analysis model on a given dataset.

    Args:
        X (list): A list of texts.
        y (list): A list of labels corresponding to the texts in X.
        model_name (str): The name of the model to be saved.
        use_mps (bool): If True, the model will use the MPS device for faster training.
    """
    if use_mps:

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(torch.device("mps"))
        training_args = TrainingArguments(
            output_dir = 'results',
            learning_rate = 5e-5, 
            num_train_epochs = 2, 
            use_mps_device = True
        )
        X_train_tokenized = tokenizer(X, padding = True, truncation = True, max_length = 512)

    else:

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        training_args = TrainingArguments(
            output_dir = 'results',
            learning_rate = 5e-5, 
            num_train_epochs = 2
        )
        X_train_tokenized = tokenizer(X, padding = True, truncation = True, max_length = 512)

    
    train_dataset = Dataset(X_train_tokenized, y)
        
    trainer = Trainer(
        model,
        training_args,
        train_dataset = train_dataset
    )
    
    trainer.train()
    trainer.save_model(model_name)

def pred_bert(text, model_name = 'bert-full-train', return_score = False):
    """
    Predict the sentiment of given text(s) using a trained BERT-based sentiment analysis model.

    Args:
        text (str or list): A string or a list of strings.
        model_name (str): The name of the trained model to be used for prediction.
        return_score (bool): If True, the function returns both the predicted label and the score.

    Returns:
        If text is a string:
            A string representing the predicted sentiment ('Positive review' or 'Negative review').

        If text is a list and return_score is True:
            A tuple of two lists, where the first list contains the predicted labels and the second list contains the scores.

        If text is a list and return_score is False:
            A list of predicted labels.
    """
    best_model = BertForSequenceClassification.from_pretrained(model_name)
    
    sentiment_analysis = pipeline(
        'sentiment-analysis', 
        model = best_model, 
        tokenizer = 'bert-base-uncased', 
        truncation = True, 
        max_length = 512, 
        padding = True
    )
    
    if isinstance(text, list):
        y_pred = []
        y_score = []
        for review in text:
            result = sentiment_analysis(review)
            y_pred.append(int(result[0]["label"][-1:]))
            y_score.append(int(result[0]["score"]))

        if return_score:
            return y_pred, y_score
        
        else:
            return y_pred

    else:

        result = sentiment_analysis(text)
        sentiment = int(result[0]['label'][-1:])

        if sentiment == 1: 
            return 'Positive review'
        
        else:
            return 'Negative review'