import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from preprocess import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger("logger")

def extract_features(test_flows, labels, mode):
    y = []
    X = []
    y_dict = {"name": 3, "dtype": 4, "vendor": 5}

    for key in test_flows.value:
        flow = test_flows.value[key]

        if (key.sid, key.did) in [('0x0000', '0xffff'), ('0x0001', '0xffff'), ('0x3990', '0xffff')]:
            continue

        for i in range(0, len(flow), 4):
            X_tmp = []
            y_tmp = None

            for label in labels:
                if label[0] in [key.sid, key.did] and (label[1], label[2]) == (key.protocol, key.additional):
                    y_tmp = label[y_dict[mode]]
                    break
            else:
                logger.error(f"Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional} - 1")

            for j in range(4):
                try:
                    X_tmp.extend([
                        normalize(flow[i + j].delta_time, "delta_time"),
                        normalize(flow[i + j].direction, "direction"),
                        normalize(flow[i + j].length, "length"),
                        normalize(flow[i + j].protocol, "protocol")
                    ])
                except:
                    X_tmp.extend([0, 0, 0, 0])

            X.append(X_tmp)
            y.append(y_tmp)

    return X, y

def evaluate(test_flows, labels, mode, model_type, model):
    logger.info(f"Evaluating {mode} {model_type} model...")

    X, y = extract_features(test_flows, labels, mode)

    if model_type in ["lstm", "rnn"]:
        total_semples = len(X) - (len(X) % 4)

        X = X[:total_semples]
        y = y[:total_semples]

        y = LabelEncoder().fit_transform(y)
        X = np.array(X).reshape(int(len(X) / 4), 4, 16)
        y = y[::4]
        y = to_categorical(y, num_classes=len(np.unique(y)))
    else:
        X = np.array(X)
        y = np.array(y)

    y_pred = model.predict(X)

    make_heatmap("../result/", y, y_pred, labels, mode, model_type)
    print_score(y, y_pred, mode, model_type)

    return y_pred

def make_heatmap(path, y_true, y_pred, labels, mode, model_type):
    label_dict = {"name": 3, "dtype": 4, "vendor": 5}
    unique_labels = np.unique([label[label_dict[mode]] for label in labels])

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plt.figure(figsize=(25, 25))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{path}{mode}_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}_heatmap.png")

def print_score(y_true, y_pred, mode, model_type):
    with open(f"../result/{mode}_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}_score.txt", 'w') as out:
        for i, (p, r, f1) in enumerate(zip(
            precision_score(y_true, y_pred, average=None),
            recall_score(y_true, y_pred, average=None),
            f1_score(y_true, y_pred, average=None)
        )):
            line = f"Class {i}: Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}"
            print(line)
            out.write(line + "\n")