import numpy as np

import logging
from preprocess import *

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger("logger")

def ovo_run(X, y):
    logger.info("Running OVO...")

    X = np.array(X)
    y = np.array(y)

    model = svm.SVC(decision_function_shape='ovo')

    # y의 클래스가 1개일 경우
    if len(np.unique(y)) == 1:
        logger.info("Only 1 class in y. Skip.")
        logger.info(f"y: {y}")
        return model
    
    model.fit(X, y)

    return model

def dt_run(X, y):
    logger.info("Running Decision Tree...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10, 15, 20]
    }

    model = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)

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

    # 정말정말 다양한 params
    params = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40],
        'min_samples_split': [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40],
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

def rnn_run(X, y):
    logger.info("Running RNN...")

    num_classes = len(np.unique(y))

    X = np.array(X)
    y = np.array(y)

    batch_size = 1
    sequence_length = 4
    input_dimension = 16

    truncation(X, sequence_length)

    X = X.reshape(-1, sequence_length, input_dimension)
    y = to_categorical(y, num_classes=num_classes)

    # 다중 분류 모델
    model = Sequential()
    model.add(SimpleRNN(32, input_shape=(sequence_length, input_dimension)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=100, batch_size=batch_size)

    return model

def lstm_run(X, y):
    logger.info("Running LSTM...")

    num_classes = len(np.unique(y))

    X = np.array(X)
    y = np.array(y)

    batch_size = 1
    sequence_length = 4
    input_dimension = 16

    truncation(X, sequence_length)
    y = to_categorical(y, num_classes=num_classes)

    X = X.reshape(-1, sequence_length, input_dimension)

    # 다중 분류 모델
    model = Sequential()
    model.add(LSTM(32, input_shape=(sequence_length, input_dimension)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=100, batch_size=batch_size)

    return model

def learn(flows, labels, mode, model_type):
    logger.info(f"Creating {mode} {model_type} model...")

    y = []
    X = []

    y_dict = {"name": 3, "dtype": 4, "vendor": 5}
    model_func = {"ovo": ovo_run, "rf": rf_run, "dt": dt_run, "rnn": rnn_run}
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

    model = model_func[model_type](X, y)

    logger.info(f"Created {mode} {model_type} model.")

    return model