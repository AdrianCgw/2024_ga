# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:21:49 2024

@author: Adrian Curic 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme()
import os

doc_dir = 'Doumentation/'
try:
    os.mkdir(doc_dir)
except:
    pass

file_col = 'part-1-dataset/field_names.txt'
filetrain = 'part-1-dataset/breast-cancer.csv'
# Read and process column names
with open(file_col,'r') as f:
    cols = f.readlines()
cols = [e.replace('\n','') for e in cols]
# Read dataframe and attack column names
df = pd.read_csv(filetrain, header = None, names = cols)

# %% Data exploration
# Check na fields
print('Check na values:')
print(df.isna().sum())
print(f'\n Check unique ids: {df.ID.nunique()}/{df.shape[0]}')
# Set boolean Y column
df['is_M'] = df.diagnosis == 'M'
print(df['is_M'].sum())

# Inspect the column types values - all columns are numerical, no need for preprocessing
print(df.dtypes)

if False:
    print(df.describe())
    for col in df.columns[2:]:
        print(df[col].describe())
    
# %% Diagnosis distribution

x = df.diagnosis.value_counts().reset_index()
fig, ax = plt.subplots(figsize=(16,9))
x.replace({'M':'Malignant','B':'Benign'}, inplace = True)
plt.pie(x['diagnosis'], labels=  x['index'], autopct='%.0f%%') 
plt.title('Diagnosis percentage')
plt.savefig(doc_dir + 'diagnosis_pie.png')
plt.show()
    
# %% Print overlapping histograms for all numerical values

col = 'smoothness_mean'
def plot_histo_diff(df, col):
    df1 = df[df.is_M]
    df0 = df[~df.is_M]
    dfg = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({col: df1[col], 'group': 'malignant'}),
        pd.DataFrame.from_dict({col: df0[col], 'group': 'benign'})
    ])
    fig, ax = plt.subplots(figsize=(16,9))
    sns.histplot(
        dfg, x= col,  hue="group", element="step",
        stat="density", common_norm=False, kde = True,
    )
    plt.title(f'Malignant vs benign value histograms for feature: {col}')
    plt.savefig(doc_dir + f'mb_histogram_{col}.png')
    plt.show()

plot_histo_diff(df, col)

# %% From visual inspection the compactness_mean STDs are much larger so the malignant/benign statistical difference will be smaller 
# for compactness_mean

col = 'compactness_mean'
plot_histo_diff(df, col)

# %% Statistical analysis for hypothesis testing: run ttest on smoothness_mean for malignant and benign
# The p_value (6.869e-29) is less than alpha (0.05) and thus we must reject the null hypothesis: meaning that diagnosis 
# is dependent on smoothness_mean

import pingouin 

def ttest_col(df, col):
    print(f'\nTtest for {col}')
    df1 = df[df.is_M]
    df0 = df[~df.is_M]
    res = pingouin.ttest(df1[col], df0[col], alternative = "two-sided") # "more" "two-sided"
    #print(res.to_string())
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(res)

col = 'smoothness_mean'
ttest_col(df, col)
col = 'compactness_mean'
ttest_col(df, col)

# %% We can see from the easy test that compactness_mean varies less that smoothness_mean between malignant and benign
# This corresponds to the p_values results (6.869e-29 for smoothness_mean, 6.342e-12 for  compactness_mean).
# p_value for compactness_mean is larger but still statistically significant.

# For an easier to digest result, we can compute the mean distances over STD:
# x = 2 * abs(s1_mean - s2_mean) / (s1_std + s2_std)
# However these values are not statisticaly correct over small samples

col = 'smoothness_mean'

def easy_test_col(df,col):
    print(f'\nEasy test for {col}')
    df1 = df[df.is_M]
    df0 = df[~df.is_M]
    x1_mean = df1[col].mean()
    x0_mean = df0[col].mean()
    x1_median = df1[col].median()
    x0_median = df0[col].median()
    d_mean =  2 * abs(x1_mean - x0_mean) / (df1[col].std() + df0[col].std())
    d_median =  2 * abs(x1_median - x0_median) / (df1[col].std() + df0[col].std())
    print(f'Malignant mean: {x1_mean}  Benign mean: {x0_mean}')
    print(f'Difference of means over mean(std): {d_mean}')    
    
    print(f'Malignant median: {x1_median}  Benign median: {x0_median}')
    print(f'Difference of medians over mean(std): {d_median}')    

col = 'smoothness_mean'
easy_test_col(df, col)
col = 'compactness_mean'
easy_test_col(df, col)

# %% Booststrap sampling: 
# Law of large numbers: over a large number of samplings, 
# mean values have a standard distribution with a mean equal to population mean    
    
# scikit-learn bootstrap
from sklearn.utils import resample

df1 = df[df.is_M]
x1_mean = df1[col].mean()
# Do 1000 samplings of 10 samples:
m_list = []
for e in range(2000):
    boot = resample(df1[col], replace=True, n_samples= 20) #, random_state=1)
    m_list.append(boot.mean())
    
fig, ax = plt.subplots(figsize=(16,9))
sns.histplot( x = m_list, kde = True)
plt.axvline(x1_mean, color = 'red')
plt.title(f'Histogram of means for bootstrap samling of {col}')
plt.savefig(doc_dir + f'boostrap_histo.png')
plt.show()
    

# %% Print correlation heatmap

df0 = df[df.columns[2:]]
fig, ax = plt.subplots(figsize=(16,9))
heatmap = sns.heatmap(df0.corr(), vmin=-1, vmax=1, annot=True)
plt.title(f'Correlation heatmap')
plt.savefig(doc_dir + f'correlation_heatmap.png')
plt.show()

# %% Print features correlation with 
# (1) The results indicate that malignant tumors have a very complex, fractal-like shape as opposed to a smooth one. 
# This is reflected by fractal_dimension_mean and also by associated geometrical  measurements: concavity_wrost, perimeter and elongation
# See:
# https://python.plainenglish.io/why-using-bar-charts-instead-of-matrix-is-helpful-to-visualize-correlation-b68fdc143c1f
# (2) smothest_worst has a negatie correlation. Intuitive because it is the opposite of fractality.
# (3) Area and (concavity_mean) have a very low correlation meaning both tumor types are abotu the same size and roundish in shape 
# Only the concavity_sd_error and concavity_worst are correlated indicating that malignant tumors although they are round, are irregular.


import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

col_feat = df.columns[2:-1]
x = df[col_feat].corrwith(df['is_M'])
x.sort_values(inplace = True)
         
fig, ax = plt.subplots(figsize =(16,12))
norm = TwoSlopeNorm(vmin=-1, vcenter =0, vmax=1)
colors = [plt.cm.RdYlGn(norm(c)) for c in x.values]
x.plot.barh(color=colors)
plt.title(f'Feature correlation to diagnosis')
plt.savefig(doc_dir + f'correlation_barplot.png')
plt.show()
# %% LEt quickly add an xgboost 

import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def generate_sample_weight(s, as_dictionary = True):
    ''' Generate sample_weight arrays for a 2 class classification
    '''
    s = np.array(s)
    ss = sum(s)
    ls = len(s)
    cls_wgt = {0: ls/(ls-ss), 1: ls/ss }
    if as_dictionary:
        return cls_wgt    
    sample_weight = np.where(s, cls_wgt[1], cls_wgt[0])
    return sample_weight

cols_x = ['radius_mean', 'radius_sd_error', 'radius_worst',
       'texture_mean', 'texture_sd_error', 'texture_worst', 'perimeter_mean',
       'perimeter_sd_error', 'perimeter_worst', 'area_mean', 'area_sd_error',
       'area_worst', 'smoothness_mean', 'smoothness_sd_error',
       'smoothness_worst', 'compactness_mean', 'compactness_sd_error',
       'compactness_worst', 'concavity_mean', 'concavity_sd_error',
       'concavity_worst', 'concave_points_mean', 'concave_points_sd_error',
       'concave_points_worst', 'symmetry_mean', 'symmetry_sd_error',
       'symmetry_worst', 'fractal_dimension_mean',
       'fractal_dimension_sd_error', 'fractal_dimension_worst']
cols_y = ['is_M']

# Do robust scalling
scaler = RobustScaler()

X = df[cols_x].to_numpy()  
y = df[cols_y].to_numpy().reshape(-1)
X = scaler.fit_transform(X)

# %%

nodesize - minimum size of terminal nodes
maxnodes - maximum number of terminal nodes
mtry - number of variables used to build each tree (thanks @user777)

# %% Random Forest prevent overfitting 

from sklearn.model_selection import GridSearchCV
#from xgboost import cv

# max_depth: This controls how deep or the number of layers deep we will have our decision trees up to. Lower prevents overfit
# min_samples_leaf: This determines the minimum number of leaf nodes. Higher prevent overfit
# min_samples_split: This determines the minimum number of samples required to split the code. lower prevents overfit
# max_leaf_nodes: This determines the maximum number of leaf nodes. lower prevents overfit

test_params = {
 'max_depth':[2,4,8], # maximum depth of a tree. lower prevents overfitting
 'min_samples_leaf':[1,2,4,8],
 'min_samples_split':[1,2,4,8],
  'max_leaf_nodes':[10,20,40,80],
}
# Result: {'max_depth': 8, 'max_leaf_nodes': 20, 'min_samples_leaf': 1, 'min_samples_split': 2}
test_params = {
 'max_depth':[6,8,12,24], # maximum depth of a tree. lower prevents overfitting
 'min_samples_leaf':[1,2,3],
 'min_samples_split':[1,2,3],
  'max_leaf_nodes':[15,20,30],
}
#{'max_depth': 12, 'max_leaf_nodes': 30, 'min_samples_leaf': 1, 'min_samples_split': 3}
test_params = {
 'max_depth':[8,12,16], # maximum depth of a tree. lower prevents overfitting
 'min_samples_leaf':[1,2],
 'min_samples_split':[2,3,4],
  'max_leaf_nodes':[20,30,40],
}
#{'max_depth': 12, 'max_leaf_nodes': 30, 'min_samples_leaf': 1, 'min_samples_split': 3}

#model = xgb.XGBClassifier(eval_metric = 'auc')
model = RandomForestClassifier(random_state=16, class_weight = 'balanced_subsample')
model_gs = GridSearchCV(estimator = model,param_grid = test_params)
#xgb_cv = xgb.cv(test_params, 

model_gs.fit(X,y)
print(model_gs.best_params_)
rf_best_params = model_gs.best_params_


# %% Random Forest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

model = RandomForestClassifier(random_state=16, class_weight = 'balanced_subsample', **rf_best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 
cm_gmean = metrics.confusion_matrix(y_test, y_pred)
p_score = metrics.precision_score(y_test, y_pred, average='binary')
r_score = metrics.recall_score(y_test, y_pred, average='binary')
#metric_list.append({'model': 'RandomForest', 'precision':p_score, 'recall': r_score})

target_names = ['Benign', 'Malignant']
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)

plt.clf()
fig, ax = plt.subplots(figsize=(16,9))
sns.heatmap(pd.DataFrame(cm_gmean), annot=True, cmap="YlGnBu" ,fmt='g')
plt.ylabel('Actual')
plt.xlabel('Prediction')
plt.title('Confusion Matrix Random Forest')
plt.savefig(doc_dir + 'cm_radom_forest.png')
plt.show()
crep = classification_report(y_test, y_pred, target_names=target_names)
print(crep)

# %%

# Feature importance Random Forest
print(model.feature_importances_)
df_fi = pd.Series(model.feature_importances_, index = cols_x).reset_index()
df_fi.columns = ['feature','feature_importance']
df_fi.sort_values(by = 'feature_importance', ascending = False, inplace = True)
# plot
fig, ax = plt.subplots(figsize=(16,9))
sns.barplot(data = df_fi, y = 'feature', x= 'feature_importance')
    
# plt.bar()
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title('RandomForest feature importance')
plt.savefig(doc_dir + 'rf_feature_importance.png')
plt.show()


# %% XGBoost prevent overfitting 

from sklearn.model_selection import GridSearchCV
#from xgboost import cv

test_params = {
 'max_depth':[3,4,5], # maximum depth of a tree. lower prevents overfitting
 'subsample':[.3,.4,.5,.6], # ratio of the training instances. lower prevents ovefitting
 'gamma':[.15,.2,.25] # minimum loss reduction required to make a further split. larger prevents overfitting 
}
#{'gamma': 0.2, 'max_depth': 4, 'subsample': 0.5}

model = xgb.XGBClassifier(eval_metric = 'auc')
model_gs = GridSearchCV(estimator = model,param_grid = test_params)
#xgb_cv = xgb.cv(test_params, 

model_gs.fit(X,y)
print(model_gs.best_params_)
xgb_best_params = model_gs.best_params_

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

sample_weight = generate_sample_weight(y_train, False)
model = xgb.XGBClassifier(eval_metric = 'auc', **xgb_best_params)
model.fit(X_train, y_train, sample_weight=sample_weight)
y_pred = model.predict(X_test) 
y_pred_proba = model.predict_proba(X_test)[::,1]
# find the best threshold
auc = metrics.roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
th_gmean = thresholds[ix]
print('Best Gmean Threshold=%f, G-Mean=%.3f, idx=%d' % (thresholds[ix], gmeans[ix], ix))
y_pred_gmean = y_pred_proba >= th_gmean

cm = metrics.confusion_matrix(y_test, y_pred)
p_score = metrics.precision_score(y_test, y_pred, average = 'binary')
r_score = metrics.recall_score(y_test, y_pred, average = 'binary')

target_names = ['Benign', 'Malignant']
plt.clf()
fig, ax = plt.subplots(figsize=(16,9))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.ylabel('Actual')
plt.xlabel('Prediction')
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
plt.title('Confusion Matrix XGB default threshold')
plt.savefig(doc_dir + 'cm_xgb_default.png')
plt.plot()
crep = classification_report(y_test, y_pred, target_names=target_names)
print(crep)
print('auc', auc)
#print('p_Score', p_score)

# %%

y_pred_gmean = y_pred_proba >= th_gmean

cm_gmean = metrics.confusion_matrix(y_test, y_pred_gmean)
p_score = metrics.precision_score(y_test, y_pred, average = 'binary')
r_score = metrics.recall_score(y_test, y_pred, average = 'binary')

target_names = ['Benign', 'Malignant']
plt.clf()
fig, ax = plt.subplots(figsize=(16,9))
sns.heatmap(pd.DataFrame(cm_gmean), annot=True, cmap="YlGnBu" ,fmt='g')
plt.ylabel('Actual')
plt.xlabel('Prediction')
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
plt.title('Confusion Matrix XGB gmean threshold')
plt.savefig(doc_dir + 'cm_xgb_gmean.png')
plt.plot()
crep = classification_report(y_test, y_pred_gmean, target_names=target_names)
print(crep)
print('auc', auc)
#print('p_Score', p_score)


# %% Print auc 

plt.clf()
fig, ax = plt.subplots(figsize=(16,9))
sns.lineplot(x = fpr, y = tpr, errorbar = None)
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('AUC')
plt.savefig(doc_dir + 'auc.png')
#mlflow.log_figure(plt.gcf(), 'auc_default.png')    
plt.plot()

# %%

# Feature importance XGBoost
print(model.feature_importances_)
df_fi = pd.Series(model.feature_importances_, index = cols_x).reset_index()
df_fi.columns = ['feature','feature_importance']
df_fi.sort_values(by = 'feature_importance', ascending = False, inplace = True)
# plot
fig, ax = plt.subplots(figsize=(16,9))
sns.barplot(data = df_fi, y = 'feature', x= 'feature_importance')
    
# plt.bar()
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title('XGB feature importance')
plt.savefig(doc_dir + 'xgb_feature_importance.png')
plt.show()

# from xgboost import plot_importance
# plot_importance(model)

# %% Random forest using mutiple random sample for the different trees. The results are combined tmulti-vote over multiple models. 
# It is a 'bagging' type of ensemble learning
# As such is less sensitive to overfitting. Can be trained in parallel and it is easier to parametrise 

# XGB boost sequencialy trains need models to handle the miss-classifications of previous models. 
# 'Boosting' type of ensemble learning. 
# Early Boosting architecture were more prone to ovefitting but XGboost implementation hadle it though regularisation, sample and votting weights, and tree prunning. 
# Sequencial trees also handle better the data imbalance.
# It is a sequencial architecture but use of data caching partially mitigate the downside.
# IT has complex hyperparameters and requires more tunning to obtain maximum performance. 
# https://www.qwak.com/post/xgboost-versus-random-forest

# %%

# Overfitting: general:
# early stopping if the test  score goes lowers
# R1 and R2 regularisation = penalty to changing internal weights
# Dropout layers in neural networks

# %%


# Does not matter:
if False:    
    y_pred_gmean = y_pred_proba >= th_gmean
    
    cm_gmean = metrics.confusion_matrix(y_test, y_pred_gmean)
    p_score = metrics.precision_score(y_test, y_pred, average = 'binary')
    r_score = metrics.recall_score(y_test, y_pred, average = 'binary')
    
    target_names = ['Benign', 'Malignant']
    plt.clf()
    fig, ax = plt.subplots(figsize=(16,9))
    sns.heatmap(pd.DataFrame(cm_gmean), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    plt.title('Confusion Matrix XGB gmean threshold')
    plt.savefig(doc_dir + 'cm_xgb_gmean.png')
    plt.plot()
    crep = classification_report(y_test, y_pred_gmean, target_names=target_names)
    print(crep)
    print('auc', auc)
    #print('p_Score', p_score)


# %%

# To Technical Audiences
# ○ Explain the limitations of your analysis and identify possible further steps
# you could take.

'''
## Technical description:
The project's scope is to predict if cells belong to malignant or benign. The dataset used describes the visual characteristics of the cell nuclei,
obtained from digitized image of a fine needle aspirate (FNA) of abreast mass.

The model performances are very good with a recall of 98.08% and a precision of 96.23%
However the model can be improved as indicated by the comments bellow:
1. Dataset is very small. As such there is a strong possibility that the model overfits and will perform worse on new data.
2. Features are not independent. From the heatmap we can observe that concavity and smoothness are strongly correlated to most of the other features
This is reflected in the XGBoost traning which mainly uses 2 features for its predictions.
3. All input features are based on the visual mesurements of the cells. Different type of features are recommended.
4. More complex models can be trained combining both neural networks and decision based models.
5. Kfold validation can be used to more precisely measure the model performance.
6. Hyperparameter tunning can be improved with a gradient based search over a continous parameter space.
7. Model overfitting can be controled through regularisation, feature sampling and early-stopping callbacks.
8. Best balance of recall/precision can reached by threshold optimization over f-beta scores.


## Non technical project description:
The project's scope is to predict if cells belong to malignant or benign. The dataset used describes the visual characteristics of the cell nuclei,
obtained from digitized image of a fine needle aspirate (FNA) of abreast mass.
The dataset was used to train models that, based on this features, predict if the cells belong to a malignant tumor or not
The model performance is decribed by recall = TP / (TP + FN), with a 98.08% values. This means that in 1.92% cases malignant tumors escape undetected 
A secondary  performance metric is the precision = TP / (TP + FP) with a 96.23% value. This means that in 3.77% cases the model gives a false alarm, reporting a benign tumor as malignant.
Sience the detection of all malignant tumors has higher priority, increasing the recall value is preferable even at the cost of precision.

An analysis of the feature importance indicates that the principal factor that contributes to the malignant tumor identification 
is the irregularity of the cell contour with features such as  
'concavity_point_sd_error' and 'fractal_dimension_mean' indicating it.

The model performance can be imporved by using a larger dataset, additional non-visual features, more complex models 
'''

1. Code
○ Feel free to comment on style, library usage, or other improvements.
2. Methodology
○ Feel free to comment on the student's data setup, modeling methodology,
and model evaluation.
3. Conceptual Understanding
○ Finally, feel free to add any suggestions or takeaways on how the student
could continue to improve their understanding of these concepts.

    
