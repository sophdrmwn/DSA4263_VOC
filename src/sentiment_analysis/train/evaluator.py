from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def get_metrics(y_pred, y_test, results_dict, model_name, y_pred_score = None):
    # evalution metrics
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

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')