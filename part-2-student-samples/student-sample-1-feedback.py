#!/usr/bin/env python

'''
A) Overall:
Make sure the code runs before submitting it. Ask teamates or use the lesson chat group. 
Using standard naming convetions (df for dataframe, X for inputs, y outputs) avoids some errors.
Try to add additional elements as indicated in the data science pipeline.

B) General comments:
1. Document the project environment (libraries and directory structure)
2. Ensure the code runs. Expect limited feeback if the code does not run
3. Follow the data science pipeline. Mainly I want to see:
- data preprocessing (fillna, scalling, type conversions)
- data exploration (correlation, outliers, feature engineering)
- model selection. Why a particular mode was chosen
- model evaluation. Metric signification, feature importance 
- summary and possible improvements
At each step I want a short comment explaining the chosen implementation.

C) Specific code comments (search for FBACK in code)
1. Check library imports in controlled environment. Imports can vary with library version. Add pip requirements or a docker file
2. Document project structure and working directory. Do not use hardcoded locations through the project, harder to maintain.
If the input dataset is not in git, document how and where to download it.
4. Check function in IDE or online:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
Read how cross validation works. Data is split in k parts to allow for multiple different train/test splits.
5. cross_val_score tries to maximize the score. 
As such, an equivalent negative must be used for all scores where a lower value is better (eg error scores)
'''

# %%

import pandas as pd
import numpy as np
#from sklearn import LinearRegression
#from sklearn.cross_validation import cross_val_score
# FBACK 1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

# FBACK 2
param_dataset = '../data/train.csv'

# Load data
# FBACK 2
# d = pd.read_csv('../data/train.csv')
data = pd.read_csv(param_dataset)

# Setup data for prediction
x1 = data.SalaryNormalized
x2 = pd.get_dummies(data.ContractType)

# Setup model
model = LinearRegression()

# Evaluate model
# FBACK 3
# from sklearn.cross_validation import cross_val_score
# from sklearn.cross_validation import train_test_split
#scores = cross_val_score(model, x2, x1, cv=1, scoring='mean_absolute_error')
# FBACK 4
# FBACK 5
scores = cross_val_score(model, x2, x1, cv= 5, scoring='neg_mean_absolute_error')
print(scores.mean())


