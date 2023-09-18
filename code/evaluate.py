import logging
from preprocess import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger("logger")

def evaluate(test_flows, labels, mode, model_type, model):
    logger.info(f"Evaluating {mode} {model_type} model...")

    y = []
    X = []
    y_dict = {"name": 3, "dtype": 4, "vendor": 5}
    for key in test_flows.value:
        flow = test_flows.value[key]

        for i in range(0, len(flow), 4):
            tmp = []
            
            for j in range(4):
                try:
                    tmp += [
                        normalize(flow[i + j].delta_time, "delta_time"),
                        normalize(flow[i + j].direction, "direction"),
                        normalize(flow[i + j].length, "length"),
                        normalize(flow[i + j].protocol, "protocol")
                    ]
                except:
                    tmp += [0, 0, 0, 0]
            
            X.append(tmp)

            for label in labels:
                if label[0] == key.sid or label[0] == key.did:
                    if label[1] == key.protocol and label[2] == key.additional:
                        y.append(label[y_dict[mode]])
                        break
<<<<<<< HEAD
            else:
                if (key.sid, key.did) == ('0x0000', '0xffff'):
                    continue
                if (key.sid, key.did) == ('0xffff', '0x0000'):
                    continue
                if (key.sid, key.did) == ('0x0001', '0xffff'):
                    continue
                if (key.sid, key.did) == ('0xffff', '0x0001'):
                    continue
                if (key.sid, key.did) == ('0x3990', '0xffff'):
                    continue
                if (key.sid, key.did) == ('0xffff', '0x3990'):
                    continue
=======
                else:
                    if (key.sid, key.did) == ('0x00000000', '0x0000ffff'):
                        continue
                    if (key.sid, key.did) == ('0x0000ffff', '0x00000000'):
                        continue
                    if (key.sid, key.did) == ('0x00000001', '0x0000ffff'):
                        continue
                    if (key.sid, key.did) == ('0x0000ffff', '0x00000001'):
                        continue
                    if (key.sid, key.did) == ('0x00003990', '0x0000ffff'):
                        continue
                    if (key.sid, key.did) == ('0x0000ffff', '0x00003990'):
                        continue
>>>>>>> origin/main

            else:
                logger.error(f"Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional}")
                exit(1)

    X = np.array(X)
    y = np.array(y)

    y_pred = model.predict(X)

    make_heatmap("../result/", y, y_pred, labels, mode, model_type)

    print_score(y, y_pred, mode, model_type)

    return y_pred

def make_heatmap(path, y_true, y_pred, labels, mode, model_type):
    label_dict = {"name": 3, "dtype": 4, "vendor": 5}
    label = []

    for i in range(len(labels)):
        label.append(labels[i][label_dict[mode]])

    label = np.unique(label)

    cm = confusion_matrix(y_true, y_pred, labels=label)

    plt.figure(figsize=(25, 25))

    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label, yticklabels=label)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(path + mode + "_" + model_type + "_heatmap.png")

def print_score(y_true, y_pred, mode, model_type):
    out = open("../result/" + mode + "_" + model_type + "_score.txt", 'w')

    out.write("Accuracy: " + str(accuracy_score(y_true, y_pred)) + "\n")
    out.write("Precision: " + str(precision_score(y_true, y_pred, average=None)) + "\n")
    out.write("Recall: " + str(recall_score(y_true, y_pred, average=None)) + "\n")
    out.write("F1: " + str(f1_score(y_true, y_pred, average=None)) + "\n")
    out.close()

    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("Precision: " + str(precision_score(y_true, y_pred, average=None)))
    print("Recall: " + str(recall_score(y_true, y_pred, average=None)))
    print("F1: " + str(f1_score(y_true, y_pred, average=None)))
    