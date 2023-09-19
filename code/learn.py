import numpy as np
import time
import logging
from preprocess import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

logger = logging.getLogger("logger")

def check_single_class(y):
    if len(np.unique(y)) == 1:
        logger.info("Only 1 class in y. Skip.")
        logger.info(f"y: {y}")
        return True
    return False

def dt_run(X, y):
    logger.info("Running Decision Tree...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10, 15, 20]
    }

    model = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)

    if not check_single_class(y):
        model.fit(X, y)

    return model

def rf_run(X, y):
    logger.info("Running Random Forest...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, 30, 40],
        'min_samples_leaf': [1, 2, 3, 5, 10, 20],
        'min_samples_split': [2, 3, 5, 10]
    }

    model = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)

    # y의 클래스가 1개일 경우
    if not check_single_class(y):
        model.fit(X, y)

    return model

def rnn_lstm_generate(X, y, seq_len, input_dim, layer_type):
    num_classes = len(np.unique(y))

    total_samples = len(X) - (len(X) % seq_len) 

    X = X[:total_samples]
    y = y[:total_samples]

    X = np.array(X).reshape(int(len(X) / seq_len), seq_len, input_dim)
    y = y[::seq_len]  # 이 줄을 추가합니다.
    y = to_categorical(y, num_classes=num_classes)

    model = Sequential()
    model.add(layer_type(32, input_shape=(seq_len, input_dim)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)

    model.fit(X, y, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

    return model

def rnn_run(X, y):
    logger.info("Running RNN...")

    return rnn_lstm_generate(X, y, 4, 16, SimpleRNN)

def lstm_run(X, y):
    logger.info("Running LSTM...")

    return rnn_lstm_generate(X, y, 4, 16, LSTM)

def learn(flows, labels, mode, model_type):
    logger.info(f"Creating {mode} {model_type} model...")

    X = []
    y = []

    y_dict = {"name": 3, "dtype": 4, "vendor": 5}
    model_func = {
        "rf": rf_run, 
        "dt": dt_run, 
        "rnn": rnn_run,
        "lstm": lstm_run
    }
    
    for key in flows.value:
        flow = flows.value[key]

        skip_condition = [
            ('0x0000', '0xffff'),
            ('0x0001', '0xffff'),
            ('0x3990', '0xffff')
        ]

        if (key.sid, key.did) in skip_condition:
            continue

        for i in range(0, len(flow), 4):
            X_tmp = []
            y_tmp = None

            for label in labels:
                if label[0] in {key.sid, key.did} and (label[1], label[2]) == (key.protocol, key.additional):
                    y_tmp = label[y_dict[mode]]
                    break
            else:
                logger.error(f"Cannot find label for {key.sid}, {key.did}, {key.protocol}, {key.additional}")
            
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
            
    logger.info(f"Created {len(X)} X, {len(y)} y.")

    # y는 문자열 형태로 저장되어 있으므로, 대응되는 숫자로 변환
    label_to_index = dict(zip(np.unique(y), range(len(np.unique(y)))))
    y = np.array([label_to_index.get(i, -1) for i in y])

    # y를 one-hot encoding
    y = to_categorical(y, num_classes=len(np.unique(y)))

    model = model_func[model_type](X, y)

    logger.info(f"Created {mode} {model_type} model.")

    # 생성 시간을 포함한 이름으로 모델 저장
    model_name = f"{mode}_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    with open(f"../model/{model_name}.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved {mode} {model_type} model as {model_name}.pkl.")

    return model