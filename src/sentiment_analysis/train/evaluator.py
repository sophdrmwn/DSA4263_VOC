from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def get_metrics(y_pred, y_test, results_dict, model_name, y_pred_score = None):
    """
    Calculates and returns classification metrics such as accuracy, precision, recall, f1 score, and AUC-ROC score.
    Updates the given results_dict with the metrics for the given model_name.
    
    Args:
    y_pred (list): Predicted labels for the test set
    y_test (list): True labels for the test set
    results_dict (dict): Dictionary to store the results of different models
    model_name (str): Name of the model for which metrics are being calculated
    y_pred_score (list, optional): Predicted probabilities for the test set (default is None)
    
    Parameters:
    dict: Updated results_dict with the metrics for the given model_name
    """
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred)  
    auc = roc_auc_score(y_test, y_pred)

    if y_pred_score is not None:
        auc = roc_auc_score(y_test, y_pred_score)
    
    metrics = {"Accuracy": acc, 
               "Recall": recall, 
               "Precision": pre,
               "F1": f1, 
               "AUC-ROC": auc}
    print(metrics)

    for metric in results_dict.keys():
        results_dict[metric][model_name] = metrics[metric]
    
    return results_dict

def model_comparison(results_dict):
    """
    Ranks models based on the metrics stored in the results_dict.
    Prints the name of the model that performed the best among all the models and the rank of all models.
    
    Parameters:
    results_dict (dict): Dictionary that contains the results of different models
    """
    rank_dict = {}
    for metric in results_dict.keys():
        sorted_dict = sorted(results_dict[metric].items(), key=lambda x:x[1], reverse = True)
        for i in range(len(sorted_dict)):
            model_name = sorted_dict[i][0]
            if model_name not in rank_dict.keys():
                rank_dict[model_name] = i
            else:
                rank_dict[model_name] += i

    best_model = sorted(rank_dict.items(), key=lambda x:x[1])[0][0]

    print(best_model + ' performed the best among all the models.')
    print(rank_dict)

def plot_roc_curve(y_test, y_pred):
    """
    Plots the ROC curve for binary classification model.
    
    Parameters:
    y_pred (list): Predicted labels or probabilities for the test set
    y_test (list): True labels for the test sets:
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')