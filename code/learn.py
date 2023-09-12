import numpy as np

import logging

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger("logger")

def svm_run(X, y):
    logger.info("Running SVM...")

    X = np.array(X)
    y = np.array(y)

    model = svm.SVC()
    model.fit(X, y)

    return model

def classify_using_svm(flows, labels, mode):
    logger.info(f"Creating {mode} SVM model...")

    y = []
    X = []
    y_dict = {"name": 3, "dtype": 4, "vendor": 5}
    for key in flows.value:
        flow = flows.value[key]

        for i in range(len(flow)):
            data = flow[i]
            for label in labels:
                if label[0] == key.sid or label[0] == key.did:
                    if label[1] == key.protocol and label[2] == key.additional:
                        y.append(label[y_dict[mode]])
                        break
            else:
                print(key.sid, key.did, key.protocol, key.additional)

                exit(1)
        
            X.append([data.delta_time, data.direction, data.length])

    return svm_run(X, y)

def random_forest_run(X, y):
    logger.info("Running Random Forest...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'n_estimators': [50, 100, 150, 200, 400],
        'max_depth': [5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    model = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)

    model.fit(X, y)

    return model

def classify_using_random_forest(flows, labels, mode):
    logger.info(f"Creating {mode} RF model...")

    y = []
    X = []

    y_dict = {"name": 3, "dtype": 4, "vendor": 5}

    for key in flows.value:
        flow = flows.value[key]

        for i in range(len(flow)):
            data = flow[i]
            for label in labels:
                if label[0] == key.sid or label[0] == key.did:
                    if label[1] == key.protocol and label[2] == key.additional:
                        y.append(label[y_dict[mode]])
                        break
            else:
                print(key.sid, key.did, key.protocol, key.additional)

                exit(1)
        
        X.append([data.delta_time, data.direction, data.length])

    return random_forest_run(X, y)