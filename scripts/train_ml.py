"""
This script will be used for training our Machine Learning models. The only input argument 
it should receive is the path to our configuration file in which we define all the training
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

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


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

    # Defining the Machine Learning model we will use to train
    if config["model"] == "catboost":
        model = CatBoostClassifier()
    elif config["model"] == "lightgbm":
        model = lgb.LGBMClassifier()
    elif config["model"] == "randomforest":
        model = RandomForestClassifier()
    elif config["model"] == "xgboost":
        model = xgb.XGBClassifier()
    elif config["model"] == "ensemble":
        clf1 = lgb.LGBMClassifier(n_jobs=-1, random_state=64, learning_rate=0.1, min_child_samples=20, num_leaves=21)
        clf2 = RandomForestClassifier(n_jobs=-1, random_state=64, bootstrap=False, max_depth=100, min_samples_leaf=4, min_samples_split=10)
        clf3 = xgb.XGBClassifier(n_jobs=-1, random_state=64, gamma=0.2, learning_rate=0.2, max_depth=7, min_child_weight=7)
        clf4 = CatBoostClassifier(eval_metric="AUC", random_state=64, depth=4, learning_rate=0.06, max_leaves=31)
        model = VotingClassifier(estimators=[("LGBM", clf1), ("RandFor", clf2), ("XGB", clf3), ("CATB", clf4)], voting="soft")
    else:
        print("You have to choose a tracked model in your config file")

    # Defining the hyperparameter grid to our training model
    if config["hyperparameter_grid"] == None:
        hyperparameter_grid = {}
    else:
        hyperparameter_grid = config["hyperparameter_grid"]

    # Defining the variable with the path where some files will be saved
    path_to_save = config["path_to_save"]

    # Training the model using the previous ML model defined, with RandomizedSearchCV function
    rand_srCV = RandomizedSearchCV(model, hyperparameter_grid, scoring=["roc_auc", "accuracy"], refit="roc_auc", random_state=64)
    start1 = time()
    rand_srCV.fit(X_train, y_train)
    end1 = time()
    training_time = end1 - start1

    # Saving the trained model with joblib method
    joblib.dump(rand_srCV, os.path.join(path_to_save, "saved_model_w_joblib.pkl"))
  
    # If you want to load the model, you can use the following line code (it is commented because it is only for explanation)
    #loaded_model = joblib.load("path_to_whatever_pickle_file.pkl")

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
