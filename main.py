import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df_train = pd.read_csv('cs-training.csv', low_memory=False)
df_test = pd.read_csv('cs-test.csv', low_memory=False)
print('training data shape', df_train.shape)
print('training data information')
print(df_train.info())

numColumns = df_train.shape[1]
pd.set_option('display.max_columns', numColumns)
print(df_train.head())

print('df_test.shape', df_test.shape)
print('test data information')
print(df_test.info())
print(df_test.head())

# Stage 1: Data Cleaning

# 1.1 rename the first column and set it to be index
df_train.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
df_test.rename(columns={'Unnamed: 0':'ID'}, inplace=True)

df_train = df_train.set_index(['ID'])
df_test = df_test.set_index(['ID'])
print(df_train.head())

# 1.2 check columns that have unique value, and delete those that have unique value; if none, continue.
uniques_train = df_train.apply(pd.Series.nunique)

if any(uniques_train == 1):
    df_train = df_train.loc[:, uniques_train != 1]
    df_test = df_test.loc[:, uniques_train != 1]

# 1.3 check duplicated rows, and delete them; if none, continue.
duplicated_train = df_train.duplicated().value_counts()
duplicated_test = df_test.duplicated().value_counts()
if duplicated_train[True] != 0:
    print('Checking duplicated data for training data:')
    print( duplicated_train)
    df_train = df_train.drop_duplicates()

if duplicated_test[True] != 0:
    print('Checking duplicated data for test data:')
    print(duplicated_test)
    df_test = df_test.drop_duplicates()


# 1.4 check abnormal data
print('Summary of training data:')
print(df_train.describe())
# we can see that the minumum value of age is 0, which is not reasonable. So we replace it with median value.
print('number of training samples that have agae smaller than 18:', len(df_train.loc[df_train['age'] < 18]))
df_train.loc[df_train['age'] == 0, 'age'] = df_train['age'].median()

print('Summary of test data:')
print(df_test.describe())
# We don't have the age problem with the test dataset. Also, note that the smallest age is larger than 18.
print('number of test samples that have agae smaller than 18:', len(df_train.loc[df_train['age'] < 18]))

# 1.5. Check the null value
print('training data information')
print(df_train.info())
print('test data information')
print(df_test.info())

# the features 'MonthlyIncome' and 'NumberOfDependents' have null value for training and test data, which we have to deal with.

# For 'MonthlyIncome':
# classify the 'MonthlyIncome' of training dataset according to retire age:
working_train = df_train.loc[(df_train['age'] >= 18) & (df_train['age'] <= 65)]
senior_train = df_train.loc[(df_train['age'] > 65)]
working_train_mean = np.mean(working_train['MonthlyIncome'])
senior_train_mean = np.mean(senior_train['MonthlyIncome'])
print('monthly income at working age (training data): mean = %.3f stdv = %.3f' % (working_train_mean, np.std(working_train['MonthlyIncome'])))
print('monthly income at retire age (training data): mean = %.3f stdv = %.3f' % (senior_train_mean, np.std(senior_train['MonthlyIncome'])))

# calculate the T-test for the means of two independent samples of scores:
import scipy
stat, p = scipy.stats.ttest_ind(working_train['MonthlyIncome'].dropna(), senior_train['MonthlyIncome'].dropna(), equal_var=False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
print(df_train.info())
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
    df_train['MonthlyIncome'] = df_train['MonthlyIncome'].replace(np.nan, df_train['MonthlyIncome'].mean())
    df_test['MonthlyIncome'] = df_test['MonthlyIncome'].replace(np.nan,df_test['MonthlyIncome'].mean())

else:
    print('Different distribution (reject H0)')

    df_train['MonthlyIncome'] = df_train.apply(lambda row: working_train_mean if np.isnan(row['MonthlyIncome'])
                                                                              and row['age']<=65 else (senior_train_mean if np.isnan(row['MonthlyIncome'])
                                                                              and row['age']>65 else row['MonthlyIncome']), axis = 1)
    working_test = df_test.loc[df_test['age'] <= 65]
    senior_test = df_test.loc[(df_test['age'] > 65)]
    working_test_mean = np.mean(working_test['MonthlyIncome'])
    senior_test_mean = np.mean(senior_test['MonthlyIncome'])

    df_test['MonthlyIncome'] = df_test.apply(lambda row: working_test_mean if np.isnan(row['MonthlyIncome'])
                                                                              and row['age']<=65 else (senior_test_mean if np.isnan(row['MonthlyIncome'])
                                                                                                                           and row['age']>65 else row['MonthlyIncome']), axis = 1)
# For 'NumberOfDependents':
df_train['NumberOfDependents'].value_counts().plot(kind = 'bar')
plt.show()
# use median value to fill null value
df_train['NumberOfDependents'].fillna(df_train['NumberOfDependents'].median(), inplace=True)
df_test['NumberOfDependents'].fillna(df_test['NumberOfDependents'].median(), inplace=True)
print('train.info')
print(df_train.info())

print('test data information')
print(df_test.info())


# Stage 2. Feature Engineering
# 2.1 Feature Scaling

Y = df_train['SeriousDlqin2yrs']
X = df_train.drop(['SeriousDlqin2yrs'], axis = 1)  # remove response variable
X_test = df_test.drop(['SeriousDlqin2yrs'], axis = 1)

x_feature = X.columns
from sklearn.preprocessing import StandardScaler

sc =StandardScaler()  # initialize scaler
X[x_feature] =sc.fit_transform(X)
X_test[x_feature] = sc.fit_transform(X_test)

# 2.2 Feature Selection
# Using Pearson correlation plot to eleminate highly correlated features.
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)

import seaborn as sns
# Create correlation matrix
corr_matrix = X.corr()

sns.heatmap(corr_matrix,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.7
threshold_corr = 0.7
to_drop = [column for column in upper.columns if any(abs(upper[column]) > threshold_corr)]
print('the features that are highly correlated and need to be dropped', to_drop)

x_feature = x_feature.drop(to_drop)
X = X[x_feature]
X_test = X_test[x_feature]

corr_matrix = X.corr()
sns.set(font_scale=0.5)
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Stage3: model training and testing
# 3.1. Check the class imbalance of the data
Y.value_counts().plot(kind='bar',alpha=.30, rot =0)
plt.show()

print('X.shape:{}, Y.shape:{}'.format(X.shape, Y.shape))
n_sample = Y.shape[0]
n_pos_sample = Y[Y == 0].shape[0]
n_neg_sample = Y[Y == 1].shape[0]
print('total # of samples：{}; # of positive samples{:.2%}; # of negative samples{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
print('# of features：', X.shape[1])

#Solving Imbalance problem using SMOTE (synthetic minority oversampling technique)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X, Y = sm.fit_sample(X, Y)
print('After apply SMOTE:')
n_sample = Y.shape[0]
n_pos_sample = Y[Y == 0].shape[0]
n_neg_sample = Y[Y == 1].shape[0]
print('total # of samples：{}; # of positive samples{:.2%}; # of negative samples{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))

# 3.2 train and test models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, roc_curve, roc_auc_score
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=0)

def plot_roc_curve(fpr, tpr, auc_score, modelName):
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, linewidth=2, label='AUC = %0.2f'%auc_score)
    plt.plot([0,1],[0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive rate", fontsize=18)
    plt.legend(modelName)

def model_analysis(model, modelName):
    # train model:
    model.fit(X_train, Y_train)

    # test model:
    # for training dataset
    Y_scores_proba_train = model.predict_proba(X_train)
    Y_scores_train = Y_scores_proba_train[:, 1]
    fpr_train, tpr_train, thresh_train = roc_curve(Y_train, Y_scores_train)
    auc_score_train = roc_auc_score(Y_train, Y_scores_train)
    print("Training: AUC Score {}".format(auc_score_train))
    plot_roc_curve(fpr_train, tpr_train, auc_score_train, modelName)
    plt.show()

    # for validation dataset
    Y_scores_proba_val = model.predict_proba(X_val)
    Y_scores_val = Y_scores_proba_val[:, 1]
    fpr_val, tpr_val, thresh_val = roc_curve(Y_val, Y_scores_val)
    auc_score_val = roc_auc_score(Y_val, Y_scores_val)
    print("Validation: AUC Score {}\n".format(auc_score_val))
    plot_roc_curve(fpr_val, tpr_val, auc_score_val, modelName)
    plt.show()

    return {modelName: auc_score_val}

results = {}
# 3.1 using logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0,
                           solver="saga",
                           penalty="l1",
                           class_weight="balanced",
                           C=1.0,
                           max_iter=500)
print('using LogisticRegression model:')
result1 = model_analysis(lr, 'LogisticRegression')
results.update(result1)

# 3.2 using logistic regression model and cross validation to select optimized parameters
from sklearn.linear_model import LogisticRegressionCV
lrCV = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty='l1', solver='saga', max_iter=500, class_weight='balanced', random_state=111)
print('using LogisticRegressionCV model:')
result2 = model_analysis(lrCV, 'LogisticRegressionCV')
results.update(result2)

# 3.3 Using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300, random_state=111, max_depth=5, class_weight='balanced')
print('using RandomForestClassifier model:')
result3 = model_analysis(rfc, 'RandomForestClassifier')
results.update(result3)

def plot_feature_importances(model, modelName):
    plt.figure(figsize=(10,8))
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel('Feature importance',  fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.ylim(-1, n_features)
    plt.legend(modelName)
    plt.show()

plot_feature_importances(rfc, 'RandomForestClassifier')

# 3.4 Using GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=112)
print('using GradientBoostingClassifier model:')
result4 = model_analysis(gbc, 'GradientBoostingClassifier')
results.update(result4)

plot_feature_importances(gbc, 'GradientBoostingClassifier')


print('AUC results using various model\n', results)
import operator
print('The selected model in this project is: ', max(results.items(), key=operator.itemgetter(1))[0])
# Therefore, we choose GradientBoostingClassifier model

# Stage 4: use the selected model to predict test data
test_proba = gbc.predict_proba(X_test)
test_scores = test_proba[:, 1]
test_scores.shape
