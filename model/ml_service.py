import time
import settings
import json
import redis
import os
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
import numpy as np

import joblib
from joblib import Parallel, delayed
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# Connecting to Redis and assign to variable `db``
# Making use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID
)


# Loading your ML or DL model and assign to variable `model` through the path where is your pickle model
path_model = "./tools_to_predict/lightgbm_model.pkl"
#path_model = "./tools_to_predict/deeplearning_model.pkl"
model = joblib.load(path_model)

# Loading the encoder and standard scaler fits to variables through the paths where are your pickle fits
encoder_hot = joblib.load("./tools_to_predict/encoder_hot.pkl")
std_scaler = joblib.load("./tools_to_predict/std_scaler.pkl")


def predict(form_dict):
    """
    Get dict from the corresponding location based on the information 
    received in the HTML, then, run our ML or DL model to get predictions.

    Parameters
    ----------
    form_dict : dict
        Name for the dictionary uploaded by the user via form.

    Returns
    -------
    prediction, prediction_proba : tuple(int, float)
        Model predicted class as a integer and the corresponding probability
        prediction of target 1 as a number.
    """

    # Converting the dict into pandas and assign the respective name columns
    data = pd.DataFrame([form_dict])
    data.columns = ['FULL_NAME', 'DNI', 'CLERK_TYPE', 'PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'QUANT_ADDITIONAL_CARDS', 'POSTAL_ADDRESS_TYPE', 'SEX', 'MARITAL_STATUS', 'QUANT_DEPENDANTS', 'EDUCATION_LEVEL_1', 'STATE_OF_BIRTH', 'CITY_OF_BIRTH', 'NACIONALITY', 'RESIDENCIAL_STATE', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'FLAG_RESIDENCIAL_PHONE', 'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE', 'MONTHS_IN_RESIDENCE', 'FLAG_MOBILE_PHONE', 'FLAG_EMAIL', 'PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS', 'QUANT_BANKING_ACCOUNTS', 'QUANT_SPECIAL_BANKING_ACCOUNTS', 'PERSONAL_ASSETS_VALUE', 'QUANT_CARS', 'COMPANY', 'PROFESSIONAL_STATE', 'PROFESSIONAL_CITY', 'PROFESSIONAL_BOROUGH', 'FLAG_PROFESSIONAL_PHONE', 'PROFESSIONAL_PHONE_AREA_CODE', 'MONTHS_IN_THE_JOB', 'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 'EDUCATION_LEVEL_2', 'FLAG_HOME_ADDRESS_DOCUMENT', 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF', 'PRODUCT', 'FLAG_ACSP_RECORD', 'AGE', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3']
    
    print(data.shape)
    print(data)
    
    # Saving the pandas into a csv file that keeps inside the info_form folder (inside module "model")
    history_csv = "info_form/history.csv"
    # If csv exists I concatenet the new row, if not, I create
    if os.path.exists(history_csv):
        form_hist = pd.read_csv(history_csv)
        form_hist = pd.concat([form_hist, data])
        form_hist.to_csv(history_csv, index=False)
    else:
        data.to_csv(history_csv, index=False)

    # Dropping the features was not taking into consideration in the preprocessing stage
    cols_one_value_prev = ['FULL_NAME', 'DNI', 'CLERK_TYPE', 'QUANT_ADDITIONAL_CARDS', 'EDUCATION_LEVEL_1', 'FLAG_MOBILE_PHONE', 'FLAG_HOME_ADDRESS_DOCUMENT', 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF', 'FLAG_ACSP_RECORD']
    data.drop(columns=cols_one_value_prev, inplace=True, errors="ignore")
    
    cols_noisy_values = ["CITY_OF_BIRTH", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH", "RESIDENCIAL_PHONE_AREA_CODE", "PROFESSIONAL_CITY", "PROFESSIONAL_BOROUGH", "PROFESSIONAL_PHONE_AREA_CODE"]
    data.drop(columns=cols_noisy_values, inplace=True, errors="ignore")

    # Managing possibility of outliers and values's types
    data["QUANT_DEPENDANTS"] = data["QUANT_DEPENDANTS"].astype(int)
    data["QUANT_DEPENDANTS"] = data["QUANT_DEPENDANTS"].mask(data["QUANT_DEPENDANTS"]>20 , 1)
    
    data["RESIDENCIAL_ZIP_3"] = pd.to_numeric(data["RESIDENCIAL_ZIP_3"], errors='coerce')
    data["PROFESSIONAL_ZIP_3"] = pd.to_numeric(data["PROFESSIONAL_ZIP_3"], errors='coerce')

    # Doing it a intern copy for details purpose
    data1 = data.copy()

    # One-hot encoding
    cols_one_hot = ['PROFESSIONAL_STATE', 'RESIDENCIAL_STATE', 'POSTAL_ADDRESS_TYPE', 'OCCUPATION_TYPE', 'QUANT_CARS', 'SEX', 'FLAG_RESIDENCIAL_PHONE', 'MATE_PROFESSION_CODE', 'FLAG_EMAIL', 'FLAG_DINERS', 'PROFESSIONAL_ZIP_3', 'COMPANY', 'QUANT_BANKING_ACCOUNTS', 'FLAG_MASTERCARD', 'EDUCATION_LEVEL_2', 'PAYMENT_DAY', 'PROFESSION_CODE', 'APPLICATION_SUBMISSION_TYPE', 'NACIONALITY', 'MARITAL_STATUS', 'RESIDENCIAL_ZIP_3', 'FLAG_VISA', 'FLAG_PROFESSIONAL_PHONE', 'FLAG_AMERICAN_EXPRESS', 'RESIDENCE_TYPE', 'FLAG_OTHER_CARDS', 'STATE_OF_BIRTH', 'QUANT_SPECIAL_BANKING_ACCOUNTS', 'PRODUCT']
    data1[cols_one_hot] = data1[cols_one_hot].astype(str)

    var_encoded = pd.DataFrame(encoder_hot.transform(data1[cols_one_hot]))
    var_encoded.columns = encoder_hot.get_feature_names_out()
    data1.drop(cols_one_hot, axis=1, inplace=True)
    var_encoded.reset_index(drop=True, inplace=True)
    data1.reset_index(drop=True, inplace=True)
    data1 = pd.concat([data1, var_encoded], axis=1)

    print(data1.shape)

    # Details after one-hot encoding
    if data._get_value(0, "MATE_PROFESSION_CODE") == "18":
        data1["MATE_PROFESSION_CODE_18.0"] = 1
    else:
        data1["MATE_PROFESSION_CODE_18.0"] = 0

    # Standard Scale
    cols_scaler = ["QUANT_DEPENDANTS", "MONTHS_IN_RESIDENCE", "PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES", "PERSONAL_ASSETS_VALUE", "MONTHS_IN_THE_JOB", "AGE"]
    data1[cols_scaler] = data1[cols_scaler].astype(float)

    data1[cols_scaler] = std_scaler.transform(data1[cols_scaler].values)

    print(data1.shape)

    # Returning the prediction results
    prediction = model.predict(data1)[0]
    
    # Returning the prediction_proba results, depending if it is from ML or DL model
    if path_model.find("deep") != -1:
        prediction_proba = model.predict_proba(data1)[1]
    else:
        prediction_proba = model.predict_proba(data1)[:, 1][0]
    
    print(type(prediction))
    print(prediction)
    print(type(prediction_proba))
    print(prediction_proba)

    return prediction, prediction_proba
    

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    or DL model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:
        # Taking a new job from Redis
        _, job_data_redis = db.brpop(settings.REDIS_QUEUE)
        job_data = json.loads(job_data_redis)
        form_dict = job_data['form_dict']

        # Running your ML or DL model on the given data
        result_pred, result_pred_proba = predict(form_dict)
        
        # Storing the model prediction in a dict with the following shape:
        # {
        #  "prediction": int,
        #  "prediction_proba": float,
        # }
        output_job_data = {
            'prediction': int(result_pred),
            'prediction_proba': float(result_pred_proba)
            }

        # Storing the results on Redis using the original job ID as the key 
        # so the API can match the results it gets to the original job sent
        db.set(job_data['id'], json.dumps(output_job_data))

        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
