import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from copy import deepcopy

def average_report(reports):
    avg_report = {}
    for key in reports[0].keys():
        avg_report[key] = 100*np.mean([report[key] for report in reports])
        avg_report[key+'_std'] = 100*np.std([report[key] for report in reports])
    return avg_report


def fit_logistic_regression(X, y, group, data_random_seed=1, repeat=1):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool_)

    # normalize x
    X = normalize(X, norm='l2')

    # set random state
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
                                                   # throughout training

    accuracies = []
    y_preds = []
    y_tests = []
    groups = []
    reports = []
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test, group_train, group_test  = train_test_split(X, y, group, test_size=0.8, random_state=rng)
        
        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool_)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        reports.append({
                'accuracy': test_acc,
                'f1_score': metrics.f1_score(y_test, y_pred, average='weighted'),
                'precision': metrics.precision_score(y_test, y_pred, average='weighted'),
                'recall': metrics.recall_score(y_test, y_pred, average='weighted'),
                'auc': metrics.roc_auc_score(y_test, y_pred, average='weighted')
            })
        accuracies.append(test_acc)
        y_preds.append(np.argmax(y_pred, axis=1))
        y_tests.append(np.argmax(y_test, axis=1))
        groups.append(group_test)

    avg_report = average_report(reports)

    return y_preds, y_tests, groups, avg_report


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool_)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    y_pred_all = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool_)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool_)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
        y_pred_all.append(y_pred)
    return accuracies, y_pred_all, []
