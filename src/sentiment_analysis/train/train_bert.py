from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from transformers import pipeline, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
from ray import tune

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

def compute_metrics(p):

    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy}

def ray_hp_space(trial):

    return {
        'learning_rate': tune.grid_search([3e-5, 5e-5]),
        'num_train_epochs': tune.grid_search([2, 4])
    }

def model_init(trial):

    return BertForSequenceClassification.from_pretrained('bert-base-uncased')

def train_bert(X_train, X_test, y_train, y_test):

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

def compute_eval_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true = labels, y_pred = pred)
    recall = recall_score(y_true = labels, y_pred = pred)
    precision = precision_score(y_true = labels, y_pred = pred)
    f1 = f1_score(y_true = labels, y_pred = pred)
    auc = roc_auc_score(y_true = labels, y_pred = pred)

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1, "auc": auc}

def eval_bert(X_train, X_test, y_train, y_test, best_params):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    X_train_tokenized = tokenizer(X_train, padding = True, truncation = True, max_length = 512)
    X_test_tokenized = tokenizer(X_test, padding = True, truncation = True, max_length = 512)

    train_dataset = Dataset(X_train_tokenized, y_train)
    test_dataset = Dataset(X_test_tokenized, y_test)
    
    training_args = TrainingArguments(
        output_dir = 'results',
        learning_rate = best_params['learning_rate'], 
        num_train_epochs = best_params['num_train_epochs']
    )
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset, 
        compute_metrics = compute_eval_metrics
    )
    
    trainer.train()
    trainer.save_model('bert-tuned')
    
    trainer.evaluate()
    
def pred_bert(X_test):
    
    best_model = BertForSequenceClassification.from_pretrained('bert-tuned')
    
    sentiment_analysis = pipeline(
        "sentiment-analysis", 
        model = best_model, 
        tokenizer = "bert-base-uncased"
    )
    
    y_pred = []
    for review in X_test:
        result = sentiment_analysis(review)
        y_pred.append(int(result[0]["label"][-1:]))
    
    return y_pred