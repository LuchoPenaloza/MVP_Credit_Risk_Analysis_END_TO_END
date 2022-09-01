# TECH MANUAL: MLOPS and/or DEVOPS ENGINEER
## Introduction
This user guide is applied to the MLOps and/or DevOps, and in general, the professionals who are involved in the deployment or production stage of the ML and DL models trained and developed at the previous stage.

## Overview
This end-to-end system will be code and deploy via an API using flask framework, and redis as the communication setup between our two services: Api and Model, that you can see in the repo structure.

A brief description here:
- Module “api”: it has all the needed code to implement the communication interface between the users and our service. It uses Flask and Redis to queue tasks to be processed by our machine or deep learning model.
- Module “model”: implements the logic to get jobs from Redis and process them with our machine or deep learning model. When we get the predicted score from our model, we must encole it on Redis again so it can be delivered to the final client.

## Previous preparation before build the end-to-end system
At the previous stage, we have developed and trained several ML and DL models, get the results (ROCAUC scores, training, and inference time, among other indicators), and choose the best one, in order to do a final validation with the TEST set.

Finally, we will decide the model we want to use for our system end-to-end, and with the pickle files of the encoder and scaler preprocessing, will be copied to our MODEL module, into the “tools_to_predict” folder (meaning, to the following location : ./model/tools_to_predict).

At the end, we will have inside that folder, the following files:
- encoder_hot.pkl
- std_scaler.pkl
- deeplearning_model.pkl
- lightgbm_model.pkl

The first two will let us preprocess the information that came from the form the final client will fill into the application. The last two are the models (one for ML and one for DL) that we choose as the best options to predict the scores of our credit risk analysis system.

Take into consideration that we put two models, but at the end, we must choose ONLY ONE to generate clean statistics about the performance of that unique model.

Finally, go to line 27 and 28 of the ml_service.py and activate or deactivate the “path_model” linecode depending on the model we decide to use (either a ML or DL model).

## Front-end development
As we mentioned before, the module Api will communicate the information given by the user, to the services of our end-to-end system. For that purpose, to collect that information, it is important to develop a front-end webpage where we can ask for the final user to give their information, to have the needed data to generate the risk analysis and score we want.

The development of this front-end was made in HTML, combining CSS styles, as you could see in the Template folder inside the module Api. You will see two .html files, one for the form that the final user will fill out, and the other one for the response that our services will bring to our final user.

## Install and run
In your command prompt or your terminal, please position at the main location of your project structure (at the same level of the docker-compose.yml file), and there, build the containers for the first time using compose build, because there are two modules + redis:
```bash
$ docker-compose up --build
```

For second and the next times, run the service using only compose:
```bash
$ docker-compose up
```

Wait to your system to deploy all the services, and when the three of them are done, you can enter to your navigator app (for example google chrome), and enter to: http://localhost/

You will have access to the form and may fill it to get a credit risk assessment.

Every time you enter information and click the ASSESS button, a new prediction will be shown, and the information of the person evaluated will be stored in the history.csv file, giving us the possibility to retrain our model in the future, and to evaluate if the prediction made by our model, was finally accuracy.

When you want to close the connection, it is recommended to stop the services with these two following steps:

1.- Press this key combination: “control+c”.

2.- Wait a few seconds and then:
```bash
$ docker-compose down
```

NOTE: If you modify the code on this deployment services (whether in some dockerfile, requeriment or .py file inside the API and MODEL modules), it is important to re build the services, for that matter, please use the following:
```bash
$ docker-compose up --build -d
```

## Thanks for reading!
