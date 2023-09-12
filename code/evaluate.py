import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
                if key.sid == '0x00000000' and key.did == '0x0000ffff':
                    continue
                elif key.sid == '0x0000ffff' and key.did == '0x00000000':
                    continue
                else:
                    logger.error(f"Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional}")
                    exit(1)
        
            X.append([data.delta_time, data.direction, data.length])

    X = np.array(X)
    y = np.array(y)

    y_pred = model.predict(X)

    return y_pred

def make_heatmap(path, y_true, y_pred, labels, mode):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.savefig(path + mode + "_heatmap.png")

def print_score(y_true, y_pred, prefix):
    out = open("../result/" + prefix + "score.txt", 'w')

    out.write("Accuracy: " + str(accuracy_score(y_true, y_pred)) + "\n")
    out.write("Precision: " + str(precision_score(y_true, y_pred, average=None)) + "\n")
    out.write("Recall: " + str(recall_score(y_true, y_pred, average=None)) + "\n")
    out.write("F1: " + str(f1_score(y_true, y_pred, average=None)) + "\n")
    out.close()

    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("Precision: " + str(precision_score(y_true, y_pred, average=None)))
    print("Recall: " + str(recall_score(y_true, y_pred, average=None)))
    print("F1: " + str(f1_score(y_true, y_pred, average=None)))
    