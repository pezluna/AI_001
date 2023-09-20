import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from preprocess import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
                    X_tmp.extend(['0', '0', '0', '0'])

            X.append(X_tmp)
            y.append(y_tmp)

    return X, y

def evaluate(test_flows, labels, mode, model_type, model):
    logger.info(f"Evaluating {mode} {model_type} model...")
    
    X, y = extract_features(test_flows, labels, mode)

    if model_type == "rnn" or model_type == "lstm":
        X = np.array(X)
        y = np.array(y)

        tokenzier_X = Tokenizer(char_level=True)
        tokenzier_X.fit_on_texts([item for sublist in X for item in sublist])

        X_sequences = [tokenzier_X.texts_to_sequences(x) for x in X]
        max_length = max([len(seq) for seq in X_sequences])
        X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_length)

        tokenizer_y = Tokenizer(char_level=True)
        tokenizer_y.fit_on_texts(y)

        y_sequences = np.array(tokenizer_y.texts_to_sequences(y))

        predictions = model.predict(X_padded)
        y_pred = [np.argmax(prediction) for prediction in predictions]
        y_true = y_sequences
    else:
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y

    label_to_index = dict(zip(np.unique(y), range(len(np.unique(y)))))

    make_heatmap("../result/", y, y_pred, labels, mode, model_type, label_to_index)
    print_score(y_true, y_pred, mode, model_type)

def make_heatmap(path, y_true, y_pred, labels, mode, model_type):
    # confusion matrix 생성
    cm = confusion_matrix(y_true, y_pred)

    # confusion matrix heatmap 생성
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"{mode} {model_type} model confusion matrix")

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