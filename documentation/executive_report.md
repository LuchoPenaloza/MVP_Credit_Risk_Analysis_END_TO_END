# EXECUTIVE REPORT
## Introduction	
The fintech ecosystem has experienced rapid growth in recent years and established itself as a key actor to meet the demands and needs of financial consumers. Nowadays there are a constant need of data analysis and modeling for problems like credit risk analysis, fraud detection, customer's churn prediction or behavioral models to predict untimely payments.

Credit risk modeling is one of the most common uses of artificial intelligence within this industry, whose goal is to use financial data to predict default risk. This type of solution has very wide applicability across not only fintech but also many business sectors and industries, and could be easily adapted to any other "risk estimation" business challenges.

For that reason, this project has been developed to deal with costumer and client credit risk analysis, using MACHINE and DEEP LEARNING models, and to be able to have an end-to-end system that allows us to accurately predict repayment probability for a given loan applicant.

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

## Project stages
1.- Exploratory Data Analysis – EDA

2.- Pre-processing of original data – Feature engineering

3.- Modelling with Machine Learning and Deep Learning – Metrics to use

4.- Training models – Analysis of results

5.- End-to-end system development

If you want a deepen report about all the stages developed, you can go to the Final report, and the other user and manual guides available in the Documentation folder.

## Analysis of modelling results
|      MODELS >>>>>    | ML LightGBM | ML Random Forest | ML XGBoost | ML CatBoost | ML Ensemble | DL MLP    |
| -------------------- | ----------- | ---------------- | ---------- | ----------- | ----------- | --------- |
| Training time (s)    |    137.9361 |         977.9247 |  6131.2461 |    891.4171 |    830.5190 |  209.3122 |
| Inference time (s)   |      0.1394 |           0.1611 |     0.1549 |      0.2837 |      0.3658 |    0.2439 |
| ROC-AUC Train score  |      0.6433 |           0.6385 |     0.6420 |      0.6458 |      0.6494 |    0.6429 |
| Accuracy Train score |      0.7396 |           0.7395 |     0.7387 |      0.7406 |      0.7406 |    0.7386 |

## Conclusions
1.- Pre-processing stage is extremely important.

2.- Final ROC-AUC metric depends a lot on the data source available.

3.- Be open to Machine and Deep Learning models in the development.

4.- The importance of ROC-AUC score and inference time in our selected final model, that finally was the LightGBM Machine Learning Model

5.- The results of our LightGBM trained model in the TEST dataset were the following:
|      ML MODEL >>>    | LightGBM |
|----------------------|----------|
| ROC-AUC Test score   |  0.6432  |
| Accuracy Test score  |  0.7405  |

6.- The importance of collect data filled out in the form.

7.- The importance to adopt Agile methodologies. For this project, the SCRUM methodology was used.

## Use cases
1.- Adaptation capacity of our risk estimation MVP to new data sources and any other industries.

## Next steps
1.- Deepen into the pre-processing stage.

2.- Continue iterating more hyperparameters and new models.

3.- Implement unitests and integration test.

4.- Develop stress test to our end-to-end system.

5.- Use authentication methods to secure our service.

6.- Implement in our model the ability to retrain with the new data from the forms filled out.

## Thanks for reading!
