import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from preprocess import *

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical


logger = logging.getLogger("logger")

def evaluate(test_flows, labels, mode, model_type, model):
    logger.info(f"Evaluating {mode} {model_type} model...")

    if model_type == "rnn" or model_type == "lstm":
        X, y = extract_features_rnn_lstm(test_flows, labels, mode)
        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)

        unique_y = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_y)}
        y = np.array([label_map[label] for label in y])

        y = to_categorical(y, num_classes=len(unique_y))
        
        y_pred = model.predict(X).argmax(axis=1)
        y_true = y.argmax(axis=1)

        logger.debug(f"y_pred: {y_pred[:10]}")
        logger.debug(f"y_true: {y_true[:10]}")
    else:
        X, y = extract_features(test_flows, labels, mode)
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y

    make_heatmap("../result/", y_true, y_pred, labels, mode, model_type)
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