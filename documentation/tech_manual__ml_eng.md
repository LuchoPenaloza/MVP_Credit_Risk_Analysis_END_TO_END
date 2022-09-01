# TECH MANUAL: ML ENGINEER
## Introduction
This user guide is applied to the ML Engineers and/or Data Scientists, and in general, the professionals who are involved in the data engineering, featuring engineering, data pre and processing, and the modelling and model trainings. The phases, in the correct sort, are the following:

## Clone the repo
First, please clone the repo (recommended to do a fork and ask for permissions if needed), accessing from your command prompt or terminal, and:
```bash
$ git clone <code_from_github>
```

Also, remember that there are some folders ignored for git, so, at this stage, it is important to manually create the folders you need to have the same structure of the repository.

## Build and run the container located in: ./docker/Dockerfile
In your command prompt or your terminal, please position at the main location of your project structure, and there, build the container for the first time (in this case we are naming it as: “mvp_credit_risk_analysis”):
```bash
$ docker build -t mvp_credit_risk_analysis -f docker/Dockerfile .
```

After the build, you can run the container whenever you want, and it must be run always before executing any script. Use the following linecode at the main location of your project in your terminal:
```bash
$ docker run --rm -it -p 8888:8888 -v $(pwd):/home/app/src mvp_credit_risk_analysis bash
$ cd src/
```
The cd src/ linecode is made to enter to the full folder structure of your project inside your running container.

To exit from the running container, use the following:
```bash
$ exit
```

NOTE: If you modify the code on this sector of the structure (whether in the dockerfile, requeriments.txt or some .py file or script), it is important to re build the respective Dockerfile, using again the docker build code line above.

## FIRST STAGE: Executing script: download_process.py
Download the original data into the respective folder on the project structure.

In our case, our original dataset is stored in cloud, specifically in an AWS S3 bucket, so we will describe the procedure to download the data from that source:

1.- Run the container “mvp_credit_risk_analysis” as we explained before in this guide.

2.- Export the keys of our AWS S3 bucket, following the variables inside the script:
```bash
$ export AWS_ACCESS_KEY_ID = <enter_the_access_id>
$ export AWS_SECRET_ACCESS_KEY = <enter_the_access_key>
```
3.- Execute the script:
```bash
$ python scripts/download_process.py
```
4.- Wait until the command line shows active again.

Now you have your original data downloaded in the structure of your project. Remember that the script is based on a specific way to download the info from the AWS S3 bucket because of the data source I worked for this MVP.

## SECOND STAGE: Executing script: pre_processing.py
Pre-process your original data with the feature engineering needed for training your models.

For the case of this MVP, this preprocessing has their unique characteristics, that are joined into the pre_processing script that we will execute now:

1.- Run the container “mvp_credit_risk_analysis” as we explained before in this guide.

2.- Execute the script:
```bash
$ python scripts/download_process.py <path_to_your_original_dataset> <path_to_the_folder_destination_of_preprocess_data>
```
3.- In our case, those paths were the following:
- Path original dataset: /home/app/src/data/original/PAKDD2010_Modeling_Data.txt
- Path folder destination: /home/app/src/data/pre_process/

4.- Wait until the command line shows active again.

5.- Now you have stored the pre_process data into the respective folder of your project structure.

The files generated at this stage are the following:
- encoder_hot.pkl
- std_scaler.pkl
- X_test.pkl
- X_train.pkl
- y_test.pkl
- y_train.pkl

Now, we have generated the pre-processed information, in a way our Machine Learning or Deep Learning models needs to be trained.

## THIRD STAGE: Executing script: train_ml.py
Training of our Machine Learning models, taking the pre-process data of the previous stage.

We will take some steps to do the executing of this script, that finally will generate our machine learning trained model:

1.- Inside the trained_models folder, create a subfolder with the name of the ML model you want to train.

2.- Inside the new created subfolder, copy the config_example_ml.yml file.

3.- Change the name of the copied file into config.yml

4.- Modify the content of the config.yml file, taking into consideration the ML model you want to train, and the hyperparameters you want to iterate with scikit learn randomized search cv.

NOTE: if you are using a CPU without the capacity of doing parallel processing, it is recommended not to use n_jobs = -1, and delete it from the config.yml file hyperparameters, and from the line 71, 72 and 73 in the train_ml.py script.

5.- Run the container “mvp_credit_risk_analysis” as we explained before in this guide.

6.- Execute the script:
```bash
$ python scripts/train_ml.py <path_to_the_config_yml_file>
```
7.- In our case, the path was the following: /home/app/src/trained_models/lightgbm/config.yml

8.- Wait until the command line shows active again.

9.- Now you have stored the results of your trained model into the new subfolder created before, and also generated and stored this trained model.

The files generated at this stage are the following:
- saved_model_w_joblib.pkl
- best_params.csv
- best_total_params.csv
- results.csv
- scores_train_test.csv

Now, we are in the possibility to train more ML models, and to identify the best option to carry to our final end-to-end system.

## FOURTH STAGE: Executing script: train_dl.py
Training of our Deep Learning models, taking the pre-process data of the previous stage.

We will take the same steps of the previous stage, with some details you must considerate:

1.- For the creation of the subfolder, include the word “deep” in the name. That is for our system to distinguish between a ML and a DL model. Example of subfolder name: “deep_learning_1”

2.- Inside the new created subfolder, copy the config_example_dl.yml file.

3.- Change the name of the copied file into config.yml

4.- Modify the content of the config.yml file, taking into consideration the DL model you want to train, and the hyperparameters you want to iterate with scikit learn randomized search cv.

NOTE: if you are using a CPU without the capacity of doing parallel processing, it is recommended not to use n_jobs = -1, and delete it from the line 84 in the train_dl.py script.

5.- If you want to add more dense layers to the DL model, or do some more tunning of it, you can modify the create_model function inside the code content of the train_dl script.

6.- Run the container “mvp_credit_risk_analysis” as we explained before in this guide.

7.- Execute the script:
```bash
$ python scripts/train_dl.py <path_to_the_config_yml_file>
```
8.- In our case, the path was the following: /home/app/src/trained_models/deep_learning_1/config.yml

9.- Wait until the command line shows active again.

10.- Now you have stored the results of your trained model into the new subfolder created before, and also generated and stored this trained model.

The files generated at this stage are the following:
- saved_model_w_joblib.pkl
- best_params.csv
- best_total_params.csv
- results.csv
- scores_train_test.csv

Now, we are in the possibility to train more DL models, and to identify the best option to carry to our final end-to-end system.

## Thanks for reading!
