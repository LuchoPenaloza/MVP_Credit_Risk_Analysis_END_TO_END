# FINAL REPORT
## Introduction	
The fintech ecosystem has experienced rapid growth in recent years and established itself as a key actor to meet the demands and needs of financial consumers. Growth was fueled by increasing demand for financial services not provided by the traditional financial sector, and increased demand for digital financial services because of the COVID-19 pandemic. In the US and Latam, there are companies that have taken the lead in these industries and are growing fast and in constant need of data analysis and modeling for problems like credit risk analysis, fraud detection, customer's churn prediction or behavioral models to predict untimely payments.  

Credit risk modeling is one of the most common uses of artificial intelligence within this industry, whose goal is to use financial data to predict default risk. When a business or individual applies for a loan, the lender must evaluate whether the business or individual can reliably repay the loan principal and interest. We can train models that learn from data (such as firm information, financial statements, previous transactions, previous credit history, etc.) and can accurately predict repayment probability for a given loan applicant.

This type of solution has very wide applicability across not only fintech but also many business sectors and industries like logistics, banks, delivery apps, freight cargo companies, insurtech, etc.) and could be easily adapted to any other "risk estimation" business challenges.

For that reason, this project has been developed to deal with costumer and client credit risk analysis, using not only MACHINE LEARNING models, but also DEEP LEARNING models, and to be able to have an end-to-end system that allows us to choose the best model found between both approaches, discovering results that we will be able to analyze throughout this report.

## Overview
These are the main bullets for this project:

- Goal: develop an end-to-end system that predict credit risk
- Problem type: binary classification
- Target: 0 to GOOD client / 1 to BAD client
- GOOD client: strong possibility to repay a loan and its interests
- BAD client: strong possibility to not to repay a loan and its interests
- Main metric to make model evaluation: ROC-AUC
- Original data: private dataset of 50k samples, with target included
- Machine Learning models evaluated: LightGBM, Random Forest, XGBoost, CatBoost, Ensemble (of the previous ML models indicated)
- Deep Learning model evaluated: Multi-Layer Perceptron (MLP) with 3 dense layers

These are the scopes we will see on this report, and cover all the stages of our project:

1.- Exploratory Data Analysis – EDA

2.- Pre-processing of original data – Feature engineering

3.- Modelling with Machine Learning and Deep Learning – Metrics to use

4.- Training models – Analysis of results

5.- End-to-end system development

6.- Conclusions

7.- Use cases

8.- Next steps

Now I will present each scope in deep, anyway, if you want, you can jump directly to the conclusions, use cases and next steps; or go to the Executive report file for an executive summary of this final report.

## Exploratory Data Analysis – EDA
In this first phase, I will analyze in deep the original data which was used to train our models and to develop the credit risk assessment.

It is important to mention that the original data had several files, but we took into consideration only two of them:
- csv file: 50,000.00 samples with private information of clients and customers, including the respective target (0 and 1)
- xls file: with the description of each feature give in the csv file

The other ones were dismissed because had incomplete information, and in other cases did not have the target, and this is a very important feature because the target allow our models to learn and train from the data.

A detailed EDA is presented in the ipynb file inside the notebook folder of our project. There, I am showing a deep analysis of some general characteristics, and some numerical and categorical features, and at the end, a correlation matrix between all the features.

Some conclusions about the original data, that I will use to make the next phase, were the following:

- Original data: 52 features + Id client + Target
- Target: 74% of 0 / 26% of 1
- The correlation matrix does not show any outstanding correlation
- There are features with only one unique value in all the samples
- There are blank spaces that will be replaced with np.nan
- There are features with a huge amount of np.nan
- There are features with high cardinality
- There are features with outliers that may be input errors, or may be values that are away from the mean and mode, but they are still real values

## Pre-processing of original data – Feature engineering
In this second phase, I will consider our EDA phase, and based on that, pre-process all the original data, and turn into a data that our models can take to learn and train.

Based on our EDA phase, I will pre-process our data in this way:

1.- Replacement of all blank spaces with np.nan

2.- The following features will be dropped because:
- Does not contribute to the model training: ID_CLIENT
- Have only 1 value in all the rows, so will not generate any learning information to our model trainings: CLERK_TYPE, QUANT_ADDITIONAL_CARDS, EDUCATION_LEVEL_1, FLAG_MOBILE_PHONE, FLAG_HOME_ADDRESS_DOCUMENT, FLAG_RG, FLAG_CPF, FLAG_INCOME_PROOF, FLAG_ACSP_RECORD.
- Have high cardinality, some np.nan, and lots of messy and misspelled values that cause noise: CITY_OF_BIRTH, RESIDENCIAL_CITY, RESIDENCIAL_BOROUGH, RESIDENCIAL_PHONE_AREA_CODE, PROFESSIONAL_CITY, PROFESSIONAL_BOROUGH, PROFESSIONAL_PHONE_AREA_CODE
- It is possible to make a deep analysis on the high cardinality features, to generated categorical options to them, but for purposes of this MVP, will not be part of the scope.

3.- Managing new categorical options to some features because it is pertinent for the quantity of np.nan that have:
- New “Other” category: APPLICATION_SUBMISSION_TYPE, PROFESSION_CODE, OCCUPATION_TYPE, MATE_PROFESSION_CODE, EDUCATION_LEVEL_2
- New “N.A.” category: PROFESSIONAL_STATE

4.- Replacement of strange values, that we consider input errors: “N” in SEX feature / “XX” in STATE_OF_BIRTH feature / Values “>20” in QUANT_DEPENDANTS

5.- Apart from the strange values, there are outliers that may be managed, but at this time I will decide to keep them, because as much as they are outliers, they are still real values, so I consider them for our model to learn and train. Also, I will use standard scaler for the numerical features at the end of the preprocessing, so that will contribute to understand the outliers.

6.- Split our 50000 samples into train/test with a train size of 80%.

7.- Making imputation of np.nan into the following features:
- Scikit learn simple imputer with Mean: MONTHS_IN_RESIDENCE
- Scikit learn simple imputer with Mode (most_frequent): SEX, STATE_OF_BIRTH, RESIDENCE_TYPE, RESIDENCIAL_ZIP_3, PROFESSIONAL_ZIP_3

8.- Scikit learn one-hot encoder of all our categorical features.

9.- Scikit learn standard scaler of all our numerical features.

10.- For the steps 7, 8 and 9, remember always to do ONLY the “fit” method to the train split, and the “transform” method to the train and test split.

Finally, as we know, this pre-processing stage (that is covered by the execution of our pre-processing.py script) will give us the X-train, X-test correctly pre-process and ready to enter to the training of our models; also, we will obtain the respective y_train, y_test; and finally, the one-hot encoder fit and the standard-scaler fit ready to use in our end-to-end system.

## Modelling with Machine Learning and Deep Learning – Metrics to use
In this third phase, I will mention some important characteristics of the Machine Learning models and the Deep Learning model were used to learn and train from our pre-process data.

It is important to mention that, for all the ML and DL cases, I used the RandomizedSearchCV model selection from the scikit learn tools, because allows us to do a random search of hyperparameters for each of our models, to choose the best combination of them.

Considering that our prediction objective is a binary classification model, to evaluate which will be the best combination in the search for hyperparameters, I decided to use the ROC-AUC metric, because it is widely used to evaluate the performance of algorithms that have possible options between two classes or categories, in our case 1 and 0. This metric takes into account the four possible outcomes for a test binary prediction: true positive, false positive, true negative, and false negative; and using a variety of thresholds (remember that the default threshold is 0.5) to plot the true positive rate against the false positive rate for a single classifier, producing the ROC curve, and with that finally calculate the area under the curve, meaning the ROC-AUC metric.

As we can see, this metric is very robust since it considers, for its calculation, a variety of indicators used in binary classification algorithms, which is why it ends up being the most used for this type of problem. As a secondary metric, I will use the accuracy metric, only for information purposes, because it could be a little tricky taking into consideration that a credit risk modelling tends to have unbalanced data.

## Training models – Analysis of results
In this fourth phase, I will describe each of the ML and DL models that I trained, the hyperparameter grid that I gave to our RandomizedSearchCV model selection, and finally, the tabulated results (obtaining thanks to the execution of our train_ml.py and train_dl.py scripts), to analyze and choose the best option to carry to our end-to-end system.

About the hyperparameter grids I will show next, the values between two asterisks (** value **) are the ones that our RandomizedSearchCV selected as the best combination that give us the best ROC-AUC result.

- Machine Learning models evaluated: LightGBM, Random Forest, XGBoost, CatBoost, Ensemble (of the previous ML models indicated)
- Deep Learning model evaluated: Multi-Layer Perceptron (MLP) with 3 dense layers

1.- ML model: LightGBM

hyperparameter_grid:

    n_jobs: [**-1**]

    random_state: [**64**]

    learning_rate: [0.001, 0.01, **0.1**, 0.5, 1]

    num_leaves: [11, 21, **31**, 51, 71]

    min_child_samples: [10, 15, 20, 25, **30**]

2.- ML model: Random Forest

hyperparameter_grid:

    n_jobs: [**-1**]

    random_state: [**64**]

    bootstrap: [True, **False**]

    max_depth: [10, 50, **100**, None]

    min_samples_leaf: [1, 2, **4**]

    min_samples_split: [**2**, 5, 10]

3.- ML model: XGBoost

hyperparameter_grid:

    n_jobs: [**-1**]

    random_state: [**64**]

    learning_rate: [0.05, 0.10, **0.2**, 0.3]

    max_depth: [3, **7**, 12, 19]

    min_child_weight: [1, 3, 5, **7**]

    gamma: [0, **0.2**, 0.4]

4.- ML model: CatBoost

hyperparameter_grid:

    eval_metric: [**"AUC"**]

    random_state: [**64**]

    depth: [**4**, 6, 8, 10]

    learning_rate: [0.01, 0.03, **0.06**, 0.1]

    max_leaves: [16, **31**, 46]

5.- ML model: Ensemble

This model is an ensemble of the previous four ML models described, using the best hyperparameters founded, and a “soft” voting classifier, as we show next:
```
clf1 = lgb.LGBMClassifier(n_jobs=-1, random_state=64, learning_rate=0.1, min_child_samples=20, num_leaves=21)
clf2 = RandomForestClassifier(n_jobs=-1, random_state=64, bootstrap=False, max_depth=100, min_samples_leaf=4, min_samples_split=10)
clf3 = xgb.XGBClassifier(n_jobs=-1, random_state=64, gamma=0.2, learning_rate=0.2, max_depth=7, min_child_weight=7)
clf4 = CatBoostClassifier(eval_metric="AUC", random_state=64, depth=4, learning_rate=0.06, max_leaves=31)
model = VotingClassifier(estimators=[("LGBM", clf1), ("RandFor", clf2), ("XGB", clf3), ("CATB", clf4)], voting="soft")
```

So, for that reason, this model has not a hyperparameter grid.

6.- DL model: Multi-Layer Perceptron (MLP)

model:

    neurons: 30

    activation_mode: "relu"

    dropout_rate: 0

    loss_type: "binary_crossentropy"

    optimizer_type: "Adam"

    callback_mode: "Activate"

data:

    batch_size: [**500**, 1000]

    epochs: [**30**, 40]

    optimizer__learning_rate: [**0.0001**]

    random_state: [**64**]

For the case of our MLP, I have two grids, that are applied to our DL model and our RandomizedSearchCV function. In case you want to change the structure of our DL model (quantity of dense layers, add more than one activation function, among others), you must go to the code on the train_dl.py script, and change this part of the code:
```
def create_model():
        dl_model = Sequential()
        dl_model.add(Dense(config["model"]["neurons"], input_shape=(X_train.shape[1],), activation=config["model"]["activation_mode"]))
        dl_model.add(Dense(config["model"]["neurons"]/2, activation=config["model"]["activation_mode"]))
        dl_model.add(Dropout(config["model"]["dropout_rate"]))
        dl_model.add(Dense(1, activation='sigmoid'))
        return dl_model
```

Of course, this is one way of methodology for a DL model, you can change other zones of the code to do more finetuning if you want.

7.- Analysis of RESULTS
Having describe all the characteristics of the models I have trained, now I will present the results of each of one. Remember that our train scripts give us the following files:
- saved_model_w_joblib.pkl
- best_params.csv
- best_total_params.csv
- results.csv
- scores_train_test.csv

From these files I catch the results, metric scores, best hyperparameter combinations, among others, that will be shown next:
|      MODELS >>>>>    | ML LightGBM | ML Random Forest | ML XGBoost | ML CatBoost | ML Ensemble | DL MLP    |
| -------------------- | ----------- | ---------------- | ---------- | ----------- | ----------- | --------- |
| Training time (s)    |    137.9361 |         977.9247 |  6131.2461 |    891.4171 |    830.5190 |  209.3122 |
| Inference time (s)   |      0.1394 |           0.1611 |     0.1549 |      0.2837 |      0.3658 |    0.2439 |
| ROC-AUC Train score  |      0.6433 |           0.6385 |     0.6420 |      0.6458 |      0.6494 |    0.6429 |
| Accuracy Train score |      0.7396 |           0.7395 |     0.7387 |      0.7406 |      0.7406 |    0.7386 |

Some important notes to consider:
- Training time: is the time that the model use to LEARN and TRAIN all over the 80% of the total of the samples (0.8*50k = 40k samples).
- Inference time: is the time that the model use to predict ONE sample, mean, one row with the information of one applicant.
- The ROC-AUC and Accuracy scores are from the train split because the metrics for the test split will be calculated only with the final model that I will choose to use in our end-to-end system.
- All the results showed are from trainings in an AWS server with an Nvidia Tesla K80 GPU, because at the end, for the deployment and production phase, it is recommended to use a server 24/7 with a good capacity of processing.

Analyzing the results, I can conclude the following:

- The fastest model is the LightGBM (ML model).
- The best ROC-AUC metric is given by the Ensemble (combination of several ML models).
- The Deep Learning model has similar results to the other Machine Learning models (it is worth to mention that after the LightGBM, it is the fastest to train).
- With respect to the metrics values, although we can notice whose models are doing better, the difference between each other is only in the order of the 3rd decimal (minimum differences).
- With respect to the training and inference time, in these parameters, we do notice an essential difference between each model, in some cases, the time is duplicated, triplicated, or even going further.
- Having said this, I will make a final decision, and choose the best model to our end-to-end system.

## End-to-end system development
This fifth phase is widely described on the technical guide and manuals, so it is recommended to check them out, because I explain how our selected model is taken to our system, and finally, it generates the predictions for the data that each new applicant fills out through the form.

It is important to mention that our end-to-end system will collect the data that the applicants will fill out thorough the form, in the following file, that is inside the model/info_form subfolder:
- history.csv

Remember that, at this phase, we must pass to our end-to-end system the following files:
- encoder_hot.pkl
- std_scaler.pkl
- selected_model.pkl

## Conclusions
1.- The pre-processing stage is extremely important, most of the time is the key to success, more in risk-estimation projects. For that reason, I took the necessary time, to analyze the available data for this project, and then make the necessary actions to pre-process that data, in order to achieve the best metrics possible.

2.- The final ROC-AUC metric depends a lot on the data source available we have available, so if we have enough and good information, and a really good data source, we will have more chances to get best possible results for our project. In this case, the data source with which I worked was not the most optimal, however, it was not an impediment to our final goal, that was to develop an end-to-end system at an MVP level, I mean, get a minimum viable product, and with that, have a solid base that afterwards can be adapted to any data source, and in consequence, if we have a better data source than the original, we automatically increase the result of our metrics.

3.- Always take into consideration approaches from Machine and Deep Learning, tech evolved every second, so we must keep updated about the innovation on traditional and actual solutions, and also the new ones; because we may have in the future better ML models, innovation on some Gradient Boosted Decision Trees, which are the most used nowadays for risk-estimation issues, or updates in DL models using TCN (temporal convolutional network) or RNN (Recurrent Neural Network) with LSTM (long short-term memory) that someday may work better than the ML approach, so it is a must to keep an eye to this evolution, because always give us the possibility to continue improve our systems and results.

4.- Always have on top of mind the importance of ROC-AUC score and inference time to analyze the results of our trainings. Having said that, the final model selected to be part of our end-to-end credit risk analysis system was the LightGBM Machine Learning Model. The arguments about this selection are the following:
- ROC-AUC metric: have the third place between all the models, but with a really short difference from the first place of only 0.006.
- Training and inference time: have the first place between all the models, with differences that can be double or triple with respect to the other models.
Remember that having a good inference time, we will have an optimal User Experience, because time is usually a delicate attribute in the production and operation phase of a system, model and/or an app.

5.- The results of our LightGBM trained model in the TEST dataset were the following:
|      ML MODEL >>>    | LightGBM |
|----------------------|----------|
| ROC-AUC Test score   |  0.6432  |
| Accuracy Test score  |  0.7405  |

6.- The importance of have new information available from the data fill by our applicants in the form. As I told you at the beginning of this part, data is the key to success in this kind of projects, so with our MVP we can collect more data from the form that our final users will fill out in the web-page, and that will allow us to enrich the original data we had, and together, the original plus the new one data, will bring us the opportunity to take next steps in order to improve our models.

7.- It is important to adopt Agile methodologies to develop this kind of projects. For this project, the SCRUM methodology was used, so please keep in mind that agile methodologies can be used to take on another challenge refer to this or any other similar projects.

## Use cases
1.- This end-to-end system can be adapted to another data source and different needs about risk estimation business challenges, because have the structure to pre-process data, refine it, learn and train Machine Learning and Deep Learning models, and be deployed in our end-to-end system. It is a matter of a correct and professional finetune of the code inside, to the specific needs to new clients and stakeholders.

2.- There are many business sectors and industries that can use our end-to-end system, because all of them have "risk estimation" business challenges. Industries such as logistics, banks, delivery apps, freight cargo companies, insurtech, among others can be take advantage of this kind of end-to-end system and generate value to their businesses.

## Next steps
1.- Deepen into the pre-processing stage, meaning, go into a deeper analysis of each feature of our original data, and make a more complex treatment in some features that required this major level of pre-processing, as, for example, high cardinality features founded. With more resources available, it will be able to improve the preprocessing and therefore the final results of our models.

2.- Continue iterating more hyperparameters and new models, because it allows us to obtain better results, and seek to improve our metrics, on the actual models and also the new ones. With more resources and better processing power we can iterate more, but is always important to establish a set and finite period of time for this phase, and therefore not fall into an infinite training loop that usually happens due to the endless number of iterations that can be obtained.

3.- Implement unitests and integration test to the main components of our end-to-end system, to build a quality assurance stage in our project.

4.- Develop stress test that will allow us to quantify the behavior of our end-to-end system when it has several prediction requests at the same time, and with that, analyze the maximum request capacity of our end-to-end system.

5.- Secured our service with an authentication method that give us a security level based on the needs of the final client or stakeholders. There are several methods such as: token-based authentication, multi-factor authentication, biometric authentication, among others.

6.- Implement in our final model the ability to retrain with new data. That is, use the data we collect from our front-end HTML, and first of all, evaluate in the near or mid-term future if the predictions of our final model for that information were accurate (because we will have the real financial behavior of the person that fill out the form, knowing, for example, if he or she pay their debts), and in second place, join the new with the original data, to have more information to finetune our models in the learning and training phase, and in consequence, improve our results.

## Thanks for reading!
