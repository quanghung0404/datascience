# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('22_workshop_data.csv')

# Drop column NA > 80 %
dataset = dataset.loc[:, dataset.isnull().sum() < 0.8*dataset.shape[0]]
# Drop colum unique < 1%
for col in dataset.columns:
    if len(dataset[col].unique()) < dataset[col].count()*0.01:
        dataset.drop(col,inplace=True,axis=1)
# Drop column unnecessary
dataset.drop('Id', axis=1, inplace=True)
y = dataset.iloc[:, -1].values
dataset.drop('SalePrice', axis=1, inplace=True)

column_dummies = ['Neighborhood','Exterior1st','Exterior2nd']
dataset = pd.get_dummies(dataset,columns = column_dummies ,drop_first=True)
column_title = dataset.columns.values.tolist()

X = dataset.iloc[:,:].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination 
import statsmodels.formula.api as sm
X = np.append(arr= np.ones((len(X), 1)).astype(int), values = X, axis = 1)
column_title_modeled = column_title
column_title_modeled.insert(0,'insert first')
index = [i for i in range(len(column_title))]
X_opt = X[:, index]

def backwardElimination(x, y, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    del column_title_modeled[j]
                    break
    regressor_OLS.summary()
    return x

SL = 0.05
X_Modeled = backwardElimination(X_opt,y, SL)

"""def backwardElimination(x,y, SL):
    numVars = len(x[0])
    temp = np.zeros((len(X),77)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
                    break
    regressor_OLS.summary()
    return x
    
SL = 0.05
X_Modeled = backwardElimination(X_opt,y, SL)"""

X_Modeled = X_Modeled[:, 1:]
del column_title_modeled[0]

X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train,y_train)
y_lasso_pred = lasso.predict(X_test)

# Ridge regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
rr = Ridge()
rr.fit(X_train, y_train)
y_ridge_pred = rr.predict(X_test)
ridge_cv = cross_val_score(estimator = rr, X = X_Modeled, y = y, cv = 10)
ridge_cv = cross_val_score(estimator = rr, X = X_test, y = y_test, cv = 10)


#Try to predict the prices, using Random Forest
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor_random_forest = RandomForestRegressor(n_estimators=20)
regressor_random_forest.fit(X_train, y_train)

# Predicting a new result
y_random_Forest_pred = regressor_random_forest.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_Modeled, y = y, cv = 10)
accuracies.mean()
accuracies.std()

rrf_cv = cross_val_score(estimator = regressor_random_forest, X = X_Modeled, y = y, cv = 10)
print(rrf_cv.mean())
print(rrf_cv.std())
rrf_cv = cross_val_score(estimator = regressor_random_forest, X = X_test, y = y_test, cv = 10)
print(rrf_cv.mean())
print(rrf_cv.std())

# Feature Importance
importance = regressor_random_forest.feature_importances_
featute_importances = zip(importance, column_title_modeled)
sorted_feature_importances = sorted(featute_importances, reverse=True)
top_15_predictors = sorted_feature_importances[0:15]
values =  [value for value, predictors in top_15_predictors]
predictors =  [predictors for value, predictors in top_15_predictors]

# Plotting
plt.figure(figsize=(15,10))
plt.title("Feature Importances")
plt.bar(range(len(predictors)), values, color="r", align="center")
plt.xticks(range(len(predictors)), predictors, rotation=90)



































# Solution---------------------------------
dataset = pd.read_csv('22_workshop_data.csv')
dataset.info()
dataset.columns
dataset= dataset.drop(["Fence", "MiscFeature", "PoolQC", "FireplaceQu", "Alley"], axis = 1)
dataset.dropna(inplace=True)

sns.distplot(dataset.SalePrice)
dataset["LogOfPrice"] = np.log(dataset.SalePrice)
dataset.drop(["SalePrice"], axis=1, inplace=True)
# modeling
y=dataset.LogOfPrice
df_temp = dataset.select_dtypes(include=["int64","float64"])
X= df_temp.drop(["LogOfPrice"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 3)

# Lineat Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

yr_hat = lr.predict(X_test)
lr.score(X_test, y_test)
lr_cv = cross_val_score(estimator = lr, X = X, y = y, cv = 5, scoring="r2")
print(lr_cv.mean())

# Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
ridge_cv = cross_val_score(estimator = ridge, X = X, y = y, cv = 5, scoring="r2")
print(ridge_cv.mean())

## Lasso
lasso = Lasso(alpha = .001)
lasso.fit(X_train,y_train)
lasso_cv = cross_val_score(estimator = lasso, X = X, y = y, cv = 5, scoring="r2")
print(lasso_cv.mean())

## Random Forest
#rrf = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=3, min_samples_leaf=5, max_features='sqrt')
rrf = RandomForestRegressor()
rrf.fit(X_train, y_train)
rrf_cv = cross_val_score(estimator = rrf, X = X, y = y, cv = 5, scoring="r2")
print(rrf_cv.mean())

# Plotting The Feature Importance
importance = rrf.feature_importances_
featute_importances = zip(importance, X.columns)
sorted_feature_importances = sorted(featute_importances, reverse=True)
top_15_predictors = sorted_feature_importances[0:15]

values =  [value for value, predictors in top_15_predictors]

predictors =  [predictors for value, predictors in top_15_predictors]

# Plotting
plt.figure(figsize=(15,10))
plt.title("Feature Importances")
plt.bar(range(len(predictors)), values, color="r", align="center")
plt.xticks(range(len(predictors)), predictors, rotation=90)
