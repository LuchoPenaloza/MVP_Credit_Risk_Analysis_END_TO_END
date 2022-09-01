"""
This script will be used for training our Deep Learning models. The only input argument it
should receive is the path to our configuration file in which we define all the training 
settings like dataset, model output folder, hyperparameters, etc.
"""
import argparse
from utils import utils
import joblib
from joblib import Parallel, delayed
import pickle
from time import time
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
import scikeras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from scikeras.wrappers import KerasClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")

    parser.add_argument(
        "config_file",
        type=str,
        help="Full path to training configuration file.",
    )

    args = parser.parse_args()

    return args


def main(config_file):
    """
    Parameters
    ----------
    config_file : str
        Full path to training configuration file.
    """

    # Loading configuration file, use of utils.load_config()
    config = utils.load_config(config_file)

    # Defining all the variables for our randomized search function
    X_train = joblib.load(config["path_X_train"])
    X_test = joblib.load(config["path_X_test"])
    y_train = joblib.load(config["path_y_train"])
    y_test = joblib.load(config["path_y_test"])
    
    # Defining the Deep Learning model with keras-tensorflow
    def create_model():
        dl_model = Sequential()
        dl_model.add(Dense(config["model"]["neurons"], input_shape=(X_train.shape[1],), activation=config["model"]["activation_mode"]))
        dl_model.add(Dense(config["model"]["neurons"]/2, activation=config["model"]["activation_mode"]))
        dl_model.add(Dropout(config["model"]["dropout_rate"]))
        dl_model.add(Dense(1, activation='sigmoid'))
        return dl_model
    
    # Activating or not the EarlyStopping callback
    if config["model"]["callback_mode"] == "Activate":
        callback_mode = [keras.callbacks.EarlyStopping(patience=2)]
    else:
        callback_mode = None
    
    # Defining the variable with the path where some files will be saved
    path_to_save = config["path_to_save"]

    # Training the model using Keras Classifier and the previous DL model defined, with RandomizedSearchCV function
    model = KerasClassifier(model=create_model(), verbose=0, loss=config["model"]["loss_type"], 
                            optimizer=config["model"]["optimizer_type"], callbacks=callback_mode)
    rand_srCV = RandomizedSearchCV(model, config["data"], n_jobs=-1, scoring=["roc_auc", "accuracy"], refit="roc_auc")
    start1 = time()
    rand_srCV.fit(X_train, y_train)
    end1 = time()
    training_time = end1 - start1

    # Saving the trained model with joblib method
    joblib.dump(rand_srCV, os.path.join(path_to_save, "saved_model_w_joblib.pkl"))
  
    # If you want to load the model, you can use the following line code (it is commented because it is only for explanation)
    ##loaded_model = joblib.load("path_to_whatever_pickle_file.pkl")

    # Obtaining the results of the training
    with open(os.path.join(path_to_save, "best_params.csv"), 'w') as f:
        for key in rand_srCV.best_params_.keys():
            f.write("%s, %s\n" % (key, rand_srCV.best_params_[key]))
    
    with open(os.path.join(path_to_save, "best_total_params.csv"), 'w') as f:
        for key in rand_srCV.best_estimator_.get_params().keys():
            f.write("%s, %s\n" % (key, rand_srCV.best_estimator_.get_params()[key]))
    
    pd.DataFrame(rand_srCV.cv_results_).to_csv(os.path.join(path_to_save, "results.csv"))

    roc_auc_train_score = rand_srCV.best_score_
    accuracy_train_score = pd.DataFrame(rand_srCV.cv_results_).loc[pd.DataFrame(rand_srCV.cv_results_)['rank_test_roc_auc'] == 1, 'mean_test_accuracy'].iloc[0]
    
    # Obtaining the results with the Test data, and comparing the results
    start2 = time()
    rand_srCV.predict(X_test.iloc[[10]])
    end2 = time()
    inference_time = end2 - start2

    y_pred_proba_1 = rand_srCV.predict_proba(X_test)[:, 1]
    y_pred = rand_srCV.predict(X_test)
    roc_auc_test_score = roc_auc_score(y_test, y_pred_proba_1)
    accuracy_test_score = accuracy_score(y_test, y_pred)

    dict_scores = {'training_time': training_time,
                   'roc_auc_train_score': roc_auc_train_score, 
                   'accuracy_train_score': accuracy_train_score, 
                   'inference_time': inference_time,
                   'roc_auc_test_score': roc_auc_test_score,
                   'accuracy_test_score': accuracy_test_score}
    
    with open(os.path.join(path_to_save, "scores_train_test.csv"), 'w') as f:
        for key in dict_scores.keys():
            f.write("%s, %s\n" % (key, dict_scores[key]))

    print("Please review your model folder")


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
