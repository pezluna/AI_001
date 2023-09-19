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

    if model_type == "rnn" or model_type == "lstm":
        # RNN, LSTM 모델의 경우, X의 길이가 4의 배수가 아닐 경우, 마지막 부분을 잘라냄
        truncate_len = len(X) % 4
        if truncate_len:
            X = X[:-truncate_len]
            y = y[:-truncate_len]
        
        # X를 4개씩 묶어서 3차원 배열로 변환
        X = np.array(X)
        X = X.reshape(-1, 4, 4)
    else:
        X = np.array(X)
        X = X.reshape(-1, 16)

    # y를 one-hot encoding
    le = LabelEncoder()
    le.fit(np.unique(labels, axis=0))
    y = le.transform(y)
    y = to_categorical(y)

    # 모델 평가
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    # heatmap 생성 및 score 출력
    label_to_index = dict(zip(le.classes_, range(len(le.classes_))))

    make_heatmap("../result/", y, y_pred, labels, mode, model_type, label_to_index)
    print_score(y, y_pred, mode, model_type)

def make_heatmap(path, y_true, y_pred, labels, mode, model_type, label_to_index):
    logger.debug(f"y_true: {y_true}")
    logger.debug(f"y_pred: {y_pred}")
    logger.debug(f"labels: {labels}")

    # y_true, y_pred를 대응되는 문자열로 변환
    index_to_label = dict(zip(label_to_index.values(), label_to_index.keys()))
    y_true = np.array([index_to_label.get(i, -1) for i in y_true])
    y_pred = np.array([index_to_label.get(i, -1) for i in y_pred])

    logger.debug(f"y_true: {y_true}")
    logger.debug(f"y_pred: {y_pred}")
    
    # y_true, y_pred를 숫자로 변환
    le = LabelEncoder()
    le.fit(np.unique(labels, axis=0))
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)

    logger.debug(f"y_true: {y_true}")
    logger.debug(f"y_pred: {y_pred}")

    # confusion matrix 생성
    cm = confusion_matrix(y_true, y_pred)
    logger.debug(f"confusion matrix: {cm}")

    # confusion matrix heatmap 생성
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)

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