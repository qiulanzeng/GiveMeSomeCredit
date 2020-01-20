# GiveMeSomeCredit
1. This is a classification problem. The goal of this project is to improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years.

2. The data is structured. To solve the problem, I used various supervised method, including logistic regression, random forest classification, and gradient boosting classification.

3. The overall workflow includes data cleaning, feature engineering, model training and validation. The details of each stage is as follows:
 
  3.1 To start with, I did data cleaning, or data pre-processing. This stage includes:
    a) rename the first column and set it to be index.
    b) check columns that have unique value, and delete those that have unique value; if none, continue.
    c) check duplicated rows, and delete them; if none, continue.
    d) check abnormal data, and did some treatment.
    e) check the null values, and fill them with reasonable values.
  
  3.2 Then, I applied feature engineering. This stage in this project mainly consists of feature scaling, Feature Selection (use Pearson correlation plot to eleminate highly correlated features)
  
  3.3 Next, I did model training and validation. Firstly, I checked the class imbalance of the data, and solving the imbalance problem using SMOTE (synthetic minority oversampling technique). Then, I split the data in 'cs-training.csv' into training and validation dataset, and trained and tested several classification model, including logistic regression, logistic regression with cross validation method to choose optimized parameters, random forest classification, and gradient boosting classification (when using the last two models, I also plot feature importances). AUC score is used as metric to select the best model. Results show that gradient boosting classification model is the best in terms of AUC score for this project: the AUC scores of the training and validation data are 0.99 and 0.98, respectively.
 
4. Finally, I used the selected model to predict test data in 'cs-test.csv'.
