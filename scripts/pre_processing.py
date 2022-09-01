import argparse
from utils import utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from joblib import Parallel, delayed
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare your dataset.")
    
    parser.add_argument(
        "data_file",
        type=str,
        help=(
            "Full path to the file having all the original dataset. E.g. "
            "`/home/app/src/data/original/PAKDD2010_Modeling_Data.txt`."
        ),
    )
    
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits pre-processed files. E.g. `/home/app/src/data/pre_process/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_file, output_data_folder):
    """
    Parameters
    ----------
    data_file : str
        Full path to the file having all the original dataset.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits pre-processed files.
    """

    # Downloading the data from the original file using "download_data" function from utils
    data = utils.download_data(data_file)

    # Replacing all blank spaces with np.nan
    data = data.replace(r'^\s*$', np.nan, regex=True)

    # Doing it a intern copy in case of manteinance
    data1 = data.copy()

    # Dropping the features that has only ONE value inside
    data1.drop(columns=data1.columns[data1.nunique() <= 1], inplace=True, errors="ignore")
    # It is important to mention that, when we acumulate a bunch of new information, we can re train again, but including some of this drop features, because will have more info inside

    # CHECKPOINT: this is going to be only for review, not to execute - for that reason it is commented
    # It will give us visibility about the cols that was dropped, in order to create the cols lists, and to manage properly
    ## data1.info()
    
    # Dropping the features that not contribute to the model training
    cols_not_contribute = ["ID_CLIENT"]
    data1.drop(columns=cols_not_contribute, inplace=True, errors="ignore")

    # Dropping the features that have high cardinality, some NaNs, and lots of messy and misspelled values that cause noise
    # We can notice that the STATE and ZIP columns has information about these deleted columns
    cols_noisy_values = ["CITY_OF_BIRTH", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH", "RESIDENCIAL_PHONE_AREA_CODE", "PROFESSIONAL_CITY", "PROFESSIONAL_BOROUGH", "PROFESSIONAL_PHONE_AREA_CODE"]
    data1.drop(columns=cols_noisy_values, inplace=True, errors="ignore")

    # Managing outliers, strange values and new categories
    data1["APPLICATION_SUBMISSION_TYPE"] = data1["APPLICATION_SUBMISSION_TYPE"].replace(["0"], "Other")
    data1["SEX"] = data1["SEX"].replace(["N"], np.nan)
    data1["QUANT_DEPENDANTS"] = data1["QUANT_DEPENDANTS"].mask(data1["QUANT_DEPENDANTS"]>20 , round(data1["QUANT_DEPENDANTS"].mean(),0))
    data1["STATE_OF_BIRTH"] = data1["STATE_OF_BIRTH"].replace(["XX"], data1["STATE_OF_BIRTH"].mode())
    data1["PROFESSIONAL_STATE"] = data1["PROFESSIONAL_STATE"].replace([np.nan], "N.A.")
    data1["PROFESSION_CODE"] = data1["PROFESSION_CODE"].replace([np.nan], "Other")
    data1["OCCUPATION_TYPE"] = data1["OCCUPATION_TYPE"].replace([np.nan], "Other")
    data1["MATE_PROFESSION_CODE"] = data1["MATE_PROFESSION_CODE"].replace([np.nan], "Other")
    data1["EDUCATION_LEVEL_2"] = data1["EDUCATION_LEVEL_2"].replace([np.nan], "Other")

    # Doing it a intern copy in case of manteinance
    data2 = data1.copy()

    # Tunning of values's types and strange values (with "coerce" errors)
    data2["RESIDENCIAL_ZIP_3"] = pd.to_numeric(data2["RESIDENCIAL_ZIP_3"], errors='coerce')
    data2["PROFESSIONAL_ZIP_3"] = pd.to_numeric(data2["PROFESSIONAL_ZIP_3"], errors='coerce')

    # CHECKPOINT: this is going to be only for review, not to execute - for that reason it is commented
    # It will give us visibility about the cols that have NaNs and their types, in order to create the cols lists, and to impute NaNs properly
    ## data2.info()

    # TRAIN/TEST Split
    X = data2.drop(["TARGET_LABEL_BAD=1"], axis=1)     # independent features from pandas dataframe
    y = data2["TARGET_LABEL_BAD=1"]                    # dependent variable from pandas dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=64, stratify=y)

    # Imputing mean to NaN values in some columns
    cols_mean = ["MONTHS_IN_RESIDENCE"]
    imputer_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer_mean.fit(X_train[cols_mean].values)
    X_train[cols_mean] = imputer_mean.transform(X_train[cols_mean].values)
    X_test[cols_mean] = imputer_mean.transform(X_test[cols_mean].values)

    # Imputing mode to NaN values in some columns
    cols_mode = ["SEX", "STATE_OF_BIRTH", "RESIDENCE_TYPE", "RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3"]
    imputer_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imputer_mode.fit(X_train[cols_mode].values)
    X_train[cols_mode] = imputer_mode.transform(X_train[cols_mode].values)
    X_test[cols_mode] = imputer_mode.transform(X_test[cols_mode].values)

    # Both lists apply to X_train and X_test
    cols_scaler = ["QUANT_DEPENDANTS", "MONTHS_IN_RESIDENCE", "PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES", "PERSONAL_ASSETS_VALUE", "MONTHS_IN_THE_JOB", "AGE"]
    cols_one_hot = list(set(X_train.columns) - set(cols_scaler))

    # One-hot encoding
    X_train[cols_one_hot] = X_train[cols_one_hot].astype(str)
    X_test[cols_one_hot] = X_test[cols_one_hot].astype(str)

    encoder_hot = OneHotEncoder(sparse=False, handle_unknown="ignore")

    encoder_hot.fit(X_train[cols_one_hot])
    # Saving encoder fit to variable through the path where will be your pickle fit
    joblib.dump(encoder_hot, os.path.join(output_data_folder, "encoder_hot.pkl"))

    var_encoded_train = pd.DataFrame(encoder_hot.transform(X_train[cols_one_hot]))
    var_encoded_train.columns = encoder_hot.get_feature_names_out()
    X_train.drop(cols_one_hot, axis=1, inplace=True)
    var_encoded_train.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_train = pd.concat([X_train, var_encoded_train], axis=1)

    var_encoded_test = pd.DataFrame(encoder_hot.transform(X_test[cols_one_hot]))
    var_encoded_test.columns = encoder_hot.get_feature_names_out()
    X_test.drop(cols_one_hot, axis=1, inplace=True)
    var_encoded_test.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.concat([X_test, var_encoded_test], axis=1)

    # Details after one-hot encoding
    ## The next two line codes apply to THIS preprocessing, because for the next one, we will have this column, so we will update (delete) them
    X_train["MATE_PROFESSION_CODE_18.0"] = 0
    X_test["MATE_PROFESSION_CODE_18.0"] = 0
    ## For next preprocessing: here we can add 0 (zero) values in the rows of the previous preprocessing, if there are new encoded columns that had not appeared before

    # To avoid index conflicts only, the same as we did in the one-hot encoding
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Standard Scale
    std_scaler = StandardScaler()

    std_scaler.fit(X_train[cols_scaler].values)
    # Saving standard scaler fit to variable through the path where will be your pickle fit
    joblib.dump(std_scaler, os.path.join(output_data_folder, "std_scaler.pkl"))

    X_train[cols_scaler] = std_scaler.transform(X_train[cols_scaler].values)
    X_test[cols_scaler] = std_scaler.transform(X_test[cols_scaler].values)

    # ANOTHER WAY: this is going to be only for review, not to execute - for that reason it is commented
    ## Saving the result files into csv
    ##X_train.to_csv("X_train.csv", index=False)
    ##X_test.to_csv("X_test.csv", index=False)
    ##y_train.to_csv("y_train.csv", index=False)
    ##y_test.to_csv("y_test.csv", index=False)

    # Saving the resulting train/test splits pre-processed files through the paths where will be your pickles files
    joblib.dump(X_train, os.path.join(output_data_folder, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(output_data_folder, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(output_data_folder, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(output_data_folder, "y_test.pkl"))


if __name__ == "__main__":
    args = parse_args()
    main(args.data_file, args.output_data_folder)
    