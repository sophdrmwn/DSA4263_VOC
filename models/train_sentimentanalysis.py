from transformers import pipeline, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
import sys

sys.path.insert(0, '../src')
from sentiment_analysis.train.train_bert import Dataset

def train_bert(X, y, model_path = 'bert-full-train', use_mps = True):
    """
    Train a BERT-based sentiment analysis model on a given dataset.

    Parameters:
        X (list): A list of texts.
        y (list): A list of labels corresponding to the texts in X.
        model_path (str): The path of the model to be saved at.
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
    trainer.save_model(model_path)