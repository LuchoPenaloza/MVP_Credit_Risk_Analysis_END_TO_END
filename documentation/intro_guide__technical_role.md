# INTRODUCTION GUIDE – TECHNICAL ROLE
## Introduction
This user guide is applied to the technical roles of the institution, meaning, machine learning engineers, data scientist, MLOps, DevOps, and in general, all the professionals involved in the IT process of the company. It will serve to know, at first hand, as an introduction on how to use the code, implement some processes, deploy the API, and other relevant stuff.

It is important to mention that this MVP project can be deploy in a CPU and AWS (GPU) environment, and of course, can be launched on web after stress-test procedures and deployment in production, that it is not the scope of this MVP, but it is part of the next steps suggestions.

## Overview of repository structure
The structure of files and folders of this project is as follows:
```
REPO_MVP_SYSTEM_END_TO_END_CREDIT_RISK_ANALYSIS
|
|
|── docker-compose.yml
|── api
|      |── __init__.py
|      |── app.py
|      |── Dockerfile
|      |── middleware.py
|      |── requirements.txt
|      |── settings.py
|      |── views.py
|      └── templates
|              |── index.html
|              └── response.html
|── model
|      |── __init__.py
|      |── Dockerfile
|      |── ml_service.py
|      |── requirements.txt
|      |── settings.py
|      |── info_form
|      |      └── history.csv
|      └── tools_to_predict
|              |── encoder_hot.pkl
|              |── std_scaler.pkl
|              |── deeplearning_model.pkl
|              └── lightgbm_model.pkl
|
|
|── data
|      |── original
|      |      |── Data.zip
|      |      |── Modeling_Data.txt
|      |      |── VariablesList.XLS
|      |      └── among_others….
|      └── pre_process
|              |── encoder_hot.pkl
|              |── std_scaler.pkl
|              |── X_test.pkl
|              |── X_train.pkl
|              |── y_test.pkl
|              └── y_train.pkl
|── docker
|      └── Dockerfile
|── notebooks
|      └── EDA.ipynb
|── scripts
|      |── download_process.py
|      |── pre_processing.py
|      |── train_dl.py
|      └── train_ml.py
|── trained_models
|      |── deep_learning_1
|      |      |── config.yml
|      |      |── saved_model_w_joblib.pkl
|      |      |── best_params.csv
|      |      |── best_total_params.csv
|      |      |── results.csv
|      |      └── scores_train_test.csv
|      |── lightgbm
|      |      |── config.yml
|      |      |── saved_model_w_joblib.pkl
|      |      |── best_params.csv
|      |      |── best_total_params.csv
|      |      |── results.csv
|      |      └── scores_train_test.csv
|      |── config_example_dl.yml
|      └── config_example_ml.yml
|── utils
|      └── utils.py
|── .dockerignore
|── .gitignore
|── requeriments.txt
|
|
|── README.md
└── documentation
        |── user_guide__final_client_ux.md
        |── intro_guide__technical_role.md
        |── tech_manual__ml_eng.md
        |── tech_manual__mlops_devops_eng.md
        |── executive_report.md
        └── final_report.md
```

## Stage directions
- In the folder structure, we can distinguish two important sectors: the API and MODEL module services, that includes the docker-compose file, and are seen on the first part of our scheme; and the ML and DL sector, that goes from data folder to before the documentation folder and are seen on the second part of our scheme.
- To follow the structure of development and how is the inner working of our system end-to-end, I recommend reading the tech manuals in this order: first ML Engineer, second MLOps / DevOps Engineer.
- For executing the linecodes you found on the tech manuals, if you need administration permission, remember to use “sudo” as a prefix to the line code you must use, but use it with precaution; and if you need to give permission to read and write in some folders, remember to use “chmod -R 777 <name_of_folder>”.

## Software and hardware requirements
For the development of this project, I use the following hardware:
- HP Pavilion x360 Convertible
- Intel Core i5-8250 CPU @ 1.60GHz 1.80GHz
- Installed RAM 4.00 GB (3.84 GB usable)

About the software use, were the following:
- Windows 10 Home Single Language / 64-bit operating system, x64 processor
- Windows Subsystem for Linux – WSL
- Ubuntu v. 20.04.4 (to start the WSL)
- Docker v. 20.10.14 (to use containers and dockerized everything)
- Git v. 2.35.1 (combined with a GitHub account)
- VSCode v. 1.69.2 (to code)

I recommend achieving at least these requirements to develop or deploy this project in a local environment. For model training purposes, I recommend the use of GPU, because there are some models that needs more than a CPU processor (for example random forest and deep learning models).

Finally, all the instances of this project were also deployed on an AWS server, to exactly take advantage of the power of its GPU processor, which was an NVIDIA TESLA K80, with 4992 Nvidia Cuda cores equipped with 12GB of RAM operational size. All the results that I will show about the model trainings in the Final Report were discovered in the AWS environment with the help of the Nvidia Tesla K80 processor.

## Thanks for reading!
