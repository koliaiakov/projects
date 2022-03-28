import pandas as pd
import numpy as np
import sys

sys.path.insert(1, "march_madness/")
from transform import *
from models import *
from data import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import argparse

parser = argparse.ArgumentParser(prog="main")
parser.add_argument(
    "--test", action="store_true", help="Quick test process with less data"
)
parser.add_argument(
    "--model",
    action="store",
    help="Indicate the type of model you want to train and test (keras_nn, gboosting, rf)",
)
parser.add_argument(
    "--mixed",
    action="store_true",
    help="choose the if tournament and regular season data have to be mixed together",
)
args = vars(parser.parse_args())  # creation of a dictionnary

data = data(mixed_model=args["mixed"], test=args["test"])

print("Build DataSet")
if data.mixed_model:
    df = data.build_mixed_model_dataset()
else:
    df = data.build_diff_model_dataset()

print("Build data for training and testing")
X_train, X_test, Y_train, Y_test = data.build_data_for_model(df)

if args["model"] == "keras_nn":
    print("Testing Keras NN model")
    model = keras_mlp(
        [(30, "relu"), (30, "relu"), (1, "sigmoid")], "binary_crossentropy"
    )

    model = keras_mlp_fit(model, X_train, Y_train, 50)
    keras_mlp_evaluate(model, X_test, Y_test)

elif args["model"] == "gboosting":
    print("Testing Sklearn GradientBoosting model")
    model = sklearn_boosting(100)
    model = sklearn_fit(model, X_train, Y_train)
    sklearn_evaluate(model, X_test, Y_test)
elif args["model"] == "rf":
    print("Testing Sklearn RandomForest model")
    model = sklearn_rf(100)
    model = sklearn_fit(model, X_train, Y_train)
    sklearn_evaluate(model, X_test, Y_test)
else:
    raise ("ERROR wrong model choice " + args["model"])
