import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

logger = logging.getLogger("logger")

def evaluate(model, test_flows, labels, mode):
    logger.info(f"Evaluating {mode} SVM model...")

    X = []
    y = []

    y_dict = {"name": 3, "dtype": 4, "vendor": 5}

    for key in test_flows.value:
        flow = test_flows.value[key]

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

    X = np.array(X)
    y = np.array(y)

    y_pred = model.predict(X)

    return y_pred

def print_score(y_true, y_pred):
    logger.info(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    logger.info(f"Confusion Matrix: {confusion_matrix(y_true, y_pred)}")