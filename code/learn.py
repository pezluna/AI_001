import numpy as np
import time
import logging
from preprocess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from bayes_opt import BayesianOptimization
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

def rnn_lstm_generate(X, y, model_type):
    # 모델 생성
    def rnn_lstm_body(X, y, model_type, num_layers, units, dropout):
        model = Sequential()

        model.add(model_type(units, input_shape=(None, num_features), return_sequences=(num_layers > 1)))
        for _ in range(num_layers - 1):
            model.add(model_type(units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(units=units, activation='relu'))
        model.add(Dense(units=len(unique_y), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def optimize_lstm(num_layers, units, dropout):
        num_layers = int(round(num_layers))
        units = int(round(units))
        model = KerasClassifier(build_fn=rnn_lstm_body, epochs=50, batch_size=4, model_type=model_type, num_layers=num_layers, units=units, dropout=dropout)

        return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    if X.shape[0] != y.shape[0]:
        logger.error("X, y shape mismatch.")
        exit(1)
    
    logger.debug(f"X shape: {X.shape}")
    logger.debug(f"y shape: {y.shape}")
    
    # y를 one-hot encoding
    unique_y = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_y)}
    y = np.array([label_map[label] for label in y])

    y = to_categorical(y, num_classes=len(unique_y))
    
    time_steps = X.shape[1]
    num_features = X.shape[2]
    units = num_features * 2

    bound = {
        'num_layers': (1, 4),
        'units': (num_features, num_features * 4),
        'dropout': (0.1, 0.3)
    }

    optimizer = BayesianOptimization(f=optimize_lstm, pbounds=bound, random_state=42)
    optimizer.maximize(init_points=5, n_iter=10)

    logger.info(f"Best hyperparameters: {optimizer.max['params']}")

    model = rnn_lstm_body(X, y, model_type, **optimizer.max['params'])

    model.fit(X, y, epochs=50, batch_size=4, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=15)])

    logger.info(f"Model summary:")
    model.summary()

    return model

def rnn_run(X, y):
    logger.info("Running RNN...")

    return rnn_lstm_generate(X, y, SimpleRNN)

def lstm_run(X, y):
    logger.info("Running LSTM...")

    return rnn_lstm_generate(X, y, LSTM)

def learn(flows, labels, mode, model_type):
    logger.info(f"Creating {mode} {model_type} model...")

    model_func = {
        "rf": rf_run, 
        "dt": dt_run, 
        "rnn": rnn_run,
        "lstm": lstm_run
    }
    if model_type in ["rf", "dt"]:
        X, y = extract_features(flows, labels, mode)

        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)
    else:
        X, y = extract_features_rnn_lstm(flows, labels, mode)

        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)

    model = model_func[model_type](X, y)

    logger.info(f"Created {mode} {model_type} model.")

    # 생성 시간을 포함한 이름으로 모델 저장
    model_name = f"{mode}_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    with open(f"../model/{model_name}.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved {mode} {model_type} model as {model_name}.pkl.")

    return model