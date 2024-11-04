from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio, true_positive_rate_difference, true_positive_rate_ratio, true_negative_rate_difference, true_negative_rate_ratio
import numpy as np

# given y_test and y_pred and groups calculate class_accuracy for all classes and groups in a 2d array
def get_class_accuracy(y_test, y_pred, groups):
    class_accuracy = []
    for group in np.unique(groups):
        group_indices = np.where(groups == group)
        group_y_test = y_test[group_indices]
        group_y_pred = y_pred[group_indices]
        accuracy = []
        for i in np.unique(y_test):
            if np.sum(group_y_test == i) == 0:
                accuracy.append(-1)
            accuracy.append(np.sum(group_y_pred == i) / np.sum(group_y_test == i))
        class_accuracy.append(accuracy)
    return class_accuracy        

def get_weights(y):
    weights = []
    for i in np.unique(y):
        weights.append(np.sum(y == i)/len(y))
    return weights

def weighted_imparity(y_test, y_pred, y, groups):
    class_accuracy = get_class_accuracy(y_test, y_pred, groups)
    weights = get_weights(y)
    imparity = 0
    for i in range(len(class_accuracy[0])):
        if class_accuracy[0][i] == -1 or class_accuracy[1][i] == -1:
            continue
        imparity += np.abs(class_accuracy[0][i] - class_accuracy[1][i]) * weights[i]
    return imparity
     
def get_fairness_metrics(y_true, y_pred, sensitive_features, labels):
    return {
        'demographic_parity_difference': demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features),
        'demographic_parity_ratio': demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features),
        'imparity': weighted_imparity(y_true, y_pred, labels, sensitive_features),
   }


# given a list of test_splits and predictions for each of them, compute the average fairness metrics
# fairness metrics are a dictionary
def get_average_fairness_metrics(y_tests, y_preds, groups, y_all):
    fairness_metrics = []
    for y_test, y_pred, group in zip(y_tests, y_preds, groups):
        fairness_metrics.append(get_fairness_metrics(y_test, y_pred, group, y_all))
    
    average_metrics = {}
    for key in fairness_metrics[0].keys():
        average_metrics[key] = np.mean([metric[key] for metric in fairness_metrics])
    
    return average_metrics