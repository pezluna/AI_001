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

    # y의 클래스가 1개일 경우
    if len(np.unique(y)) == 1:
        logger.info("Only 1 class in y. Skip.")
        logger.info(f"y: {y}")
        return model
    
    model.fit(X, y)

    return model

def classify_using_svm(flows, labels, mode):
    logger.info(f"Creating {mode} SVM model...")

    y = []
    X = []
    y_dict = {"name": 3, "dtype": 4, "vendor": 5}
    for key in flows.value:
        flow = flows.value[key]

        for i in range(0, len(flow), 4):
            tmp = []
            
            for j in range(4):
                try:
                    tmp += [flow[i + j].delta_time, flow[i + j].direction, flow[i + j].length]
                except:
                    break
            
            X.append(tmp)

            for label in labels:
                if label[0] == key.sid or label[0] == key.did:
                    if label[1] == key.protocol and label[2] == key.additional:
                        y.append(label[y_dict[mode]])
                        break
                else:
                    if (key.sid, key.did) == ('0x00000000', '0x0000ffff'):
                        continue
                    if (key.sid, key.did) == ('0x00000001', '0x0000ffff'):
                        continue
                    if (key.sid, key.did) == ('0x00003990', '0x0000ffff'):
                        continue
                
                logger.error(f"1: Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional}")
                exit(1)

            else:
                logger.error(f"2: Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional}")
                exit(1)

    return svm_run(X, y)

def random_forest_run(X, y):
    logger.info("Running Random Forest...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'n_estimators': [5, 10, 20, 50, 100, 150, 200, 400],
        'max_depth': [5, 10, 15, 20, 25],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10, 15, 20]
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

def classify_using_random_forest(flows, labels, mode):
    logger.info(f"Creating {mode} RF model...")

    y = []
    X = []
    y_dict = {"name": 3, "dtype": 4, "vendor": 5}
    for key in flows.value:
        flow = flows.value[key]

        for i in range(0, len(flow), 4):
            tmp = []
            
            for j in range(4):
                try:
                    tmp += [flow[i + j].delta_time, flow[i + j].direction, flow[i + j].length]
                except:
                    break
            
            X.append(tmp)

            for label in labels:
                if label[0] == key.sid or label[0] == key.did:
                    if label[1] == key.protocol and label[2] == key.additional:
                        y.append(label[y_dict[mode]])
                        break
                else:
                    if (key.sid, key.did) == ('0x00000000', '0x0000ffff'):
                        continue
                    if (key.sid, key.did) == ('0x00000001', '0x0000ffff'):
                        continue
                    if (key.sid, key.did) == ('0x00003990', '0x0000ffff'):
                        continue
                
                logger.error(f"1: Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional}")

            else:
                logger.error(f"2: Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional}")
                exit(1)

    return random_forest_run(X, y)