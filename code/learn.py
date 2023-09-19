import numpy as np

import logging
from preprocess import *

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger("logger")

def ovo_run(X, y):
    logger.info("Running OVO...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'C': [1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['auto']
    }

    model = GridSearchCV(svm.SVC(decision_function_shape='ovo'), params, cv=5, n_jobs=-1)

    # y의 클래스가 1개일 경우
    if len(np.unique(y)) == 1:
        logger.info("Only 1 class in y. Skip.")
        logger.info(f"y: {y}")
        return model
    
    model.fit(X, y)

    return model

def ovr_run(X, y):
    logger.info("Running OVR...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'C': [1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['auto']
    }

    model = GridSearchCV(svm.SVC(decision_function_shape='ovr'), params, cv=5, n_jobs=-1)

    # y의 클래스가 1개일 경우
    if len(np.unique(y)) == 1:
        logger.info("Only 1 class in y. Skip.")
        logger.info(f"y: {y}")
        return model

    model.fit(X, y)

    return model

def rf_run(X, y):
    logger.info("Running Random Forest...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'n_estimators': [5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400],
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10, 15, 20, 25]
    }

    model = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)

    # reshape

    # y의 클래스가 1개일 경우
    if len(np.unique(y)) == 1:
        logger.info("Only 1 class in y. Skip.")
        logger.info(f"y: {y}")
        return model

    model.fit(X, y)

    return model

def learn(flows, labels, mode, model):
    logger.info(f"Creating {mode} {model} model...")

    y = []
    X = []

    y_dict = {"name": 3, "dtype": 4, "vendor": 5}
    model_func = {"ovo": ovo_run, "ovr": ovr_run, "rf": rf_run}
    for key in flows.value:
        flow = flows.value[key]

        if (key.sid, key.did) == ('0x0000', '0xffff'):
            continue
        if (key.sid, key.did) == ('0x0001', '0xffff'):
            continue
        if (key.sid, key.did) == ('0x3990', '0xffff'):
            continue

        for i in range(0, len(flow), 4):
            X_tmp = []
            y_tmp = None

            for label in labels:
                if label[0] == key.sid or label[0] == key.did:
                    if label[1] == key.protocol and label[2] == key.additional:
                        y_tmp = label[y_dict[mode]]
                        break
            else:
                logger.error(f"Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional} - 1")
            
            for j in range(4):
                try:
                    X_tmp += [
                        normalize(flow[i + j].delta_time, "delta_time"),
                        normalize(flow[i + j].direction, "direction"),
                        normalize(flow[i + j].length, "length"),
                        normalize(flow[i + j].protocol, "protocol")
                    ]
                except:
                    X_tmp += [0, 0, 0, 0]
            
            X.append(X_tmp)
            y.append(y_tmp)
            
    logger.info(f"Created {len(X)} X, {len(y)} y.")

    model = model_func[model](X, y)

    logger.info(f"Created {mode} {model} model.")

    return model