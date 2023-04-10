from transformers import pipeline, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def train_bert_full(X, y, model_name = 'bert_full_train'):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    training_args = TrainingArguments(
        output_dir = 'results',
        learning_rate = 5e05, 
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

def pred_bert_line(text):

    best_model = BertForSequenceClassification.from_pretrained('bert_full_train')
    
    sentiment_analysis = pipeline(
        "sentiment-analysis", 
        model = best_model, 
        tokenizer = "bert-base-uncased", 
        truncation = True, 
        max_length = 512, 
        padding = True
    )
    
    result = sentiment_analysis(text)
    sentiment = int(result[0]["label"][-1:])

    if sentiment == 1: 
        return "Positive review"
    
    else:
        return "Negative review"