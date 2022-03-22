import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize


def keras_mlp(layer_params, loss, optimizer="adam", input_dim=None):
    model = Sequential()
    for (n, act_function) in layer_params:
        model.add(Dense(n, activation=act_function))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def keras_mlp_fit(model, X_train, Y_train, epochs):
    model.fit(X_train, Y_train, epochs=epochs)
    return model


def keras_mlp_evaluate(model, X_test, Y_test):
    y_pred = model.predict(X_test, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(Y_test, y_pred_bool))
    return ()


def sklearn_boosting(
    n_estimators,
    learning_rate=1.0,
    max_depth=1,
    random_state=0,
    criterion="friedman_mse",
):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        criterion=criterion,
    )
    return model


def sklearn_rf(n_estimators, criterion="gini", max_depth=5, random_state=0):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
    )
    return model


def sklearn_fit(model, X_train, Y_train):
    model.fit(X_train, Y_train)
    return model


def sklearn_evaluate(model, X_test, Y_test):
    res = model.predict(X_test)
    print(classification_report(Y_test, res))
    return ()
