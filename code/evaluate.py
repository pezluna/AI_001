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

    # 모든 레이블을 수집하여 딕셔너리를 생성합니다.
    all_labels = list(set(y))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # y의 레이블을 정수 인덱스로 변환합니다.
    y = [label_to_index[label] for label in y]

    if model_type == "rnn" or model_type == "lstm":
        # 먼저 X와 y의 길이를 맞춰줍니다.
        X = X[::4]
        y = y[::4]

        # X의 크기를 4x16의 배수로 만듭니다.
        truncate_len = len(X) % 4
        if truncate_len:
            X = X[:-truncate_len]
            y = y[:-truncate_len]

        X = np.array(X)
        X = X.reshape(len(X) // 4, 4, 16)
        y = to_categorical(y, num_classes=len(np.unique(y)))

    y_pred = model.predict(X)
    
    if model_type == "rf" or model_type == "dt":
        unique_labels = np.unique(y)
        label_to_index = dict(zip(unique_labels, range(len(unique_labels))))
        y_pred = np.array([label_to_index.get(i, -1) for i in y_pred])
    elif model_type == "rnn" or model_type == "lstm":
        y_pred = np.argmax(y_pred, axis=1)
    
    # only tested on rnn
    y = np.array(y, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    make_heatmap("../result/", y, y_pred, labels, mode, model_type, all_labels, label_to_index)
    print_score(y, y_pred, mode, model_type)

def make_heatmap(path, y_true, y_pred, labels, mode, model_type, all_labels, label_to_index):
    label_dict = {"name": 3, "dtype": 4, "vendor": 5}

    # Use label_to_index to convert indices back to original labels
    y_true_labels = [all_labels[i] for i in y_true]
    y_pred_labels = [all_labels[i] for i in y_pred]

    y_true = [labels[label_to_index[label]][label_dict[mode]] for label in y_true_labels]
    y_pred = [labels[label_to_index[label]][label_dict[mode]] for label in y_pred_labels]

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 글자 크기 조정
    sns.set(font_scale=1.5)

    plt.figure(figsize=(20, 20))

    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.savefig(f"{path}{mode}_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}_heatmap.png")

def print_score(y_true, y_pred, mode, model_type):
    with open(f"../result/{mode}_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}_score.txt", 'w') as out:
        out.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
        out.write(f"Precision: {precision_score(y_true, y_pred, average='weighted')}\n")
        out.write(f"Recall: {recall_score(y_true, y_pred, average='weighted')}\n")
        out.write(f"F1: {f1_score(y_true, y_pred, average='weighted')}\n")

    logger.info(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    logger.info(f"Precision: {precision_score(y_true, y_pred, average='weighted')}")
    logger.info(f"Recall: {recall_score(y_true, y_pred, average='weighted')}")
    logger.info(f"F1: {f1_score(y_true, y_pred, average='weighted')}")