#!/usr/bin/env python

'''

A) Overall:
Looks great. Try to add additional elements as indicated in the data science pipeline.
    
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
5. cross_val_score tries to maximize the score. 
As such, an equivalent negative must be used for all scores where a lower value is better (eg error scores)
'''

# %%

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn.cross_validation import cross_val_score
# FBACK 1
from sklearn.model_selection import cross_val_score

# FBACK 2
param_dataset = '../data/train.csv'

# Load data
data = pd.read_csv(param_dataset)


# Setup data for prediction
y = data.SalaryNormalized
X = pd.get_dummies(data.ContractType)

# Setup model
model = LinearRegression()

# Evaluate model
# FBACK 5
#scores = cross_val_score(model, X, y, cv=5, scoring='mean_absolute_error')
scores = cross_val_score(model, X, y, cv= 5, scoring='neg_mean_absolute_error')
print(scores.mean())


