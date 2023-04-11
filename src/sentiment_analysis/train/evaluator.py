from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def get_metrics(y_pred, y_test, results_dict, model_name):
    # evalution metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred)  
    auc = roc_auc_score(y_test, y_pred)

    results_dict[model_name] = {"accuracy": acc, 
                                "recall": recall, 
                                "precision": pre, 
                                "f1": f1, 
                                "auc": auc}
    
    return results_dict