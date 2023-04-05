from transformations import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
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
        'learning_rate': tune.choice([2e-5, 3e-5, 5e-5]),
        'num_train_epochs': tune.choice([3, 5])
    }

def model_init(model_name = 'bert-based-uncased'):
    return BertForSequenceClassification.from_pretrained(model_name)

def preprocess_bert(df):

    df['clean_text'] = df['Text'].apply(lambda x: get_cleantext(x))
    df['label'] = df.Sentiment.map({"positive": 1, "negative": 0})

    df_clean = df[['clean_text', 'label']]
    X = list(df_clean['clean_text'])
    y = list(df_clean['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 4263)

    return X_train, X_test, y_train, y_test

def initialise_bert(X_train, X_test, y_train, y_test, model_name = 'bert-base-uncased'):

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    X_train_tokenized = tokenizer(X_train, padding = True, truncation = True, max_length = 512)
    X_test_tokenized = tokenizer(X_test, padding = True, truncation = True, max_length = 512)

    train_dataset = Dataset(X_train_tokenized, y_train)
    test_dataset = Dataset(X_test_tokenized, y_test)

    return train_dataset, test_dataset

def train_bert(train_dataset, test_dataset, n_trials = 2):

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
        n_trials = n_trials
    )

    return best_trial
