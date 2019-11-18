# https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155


import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def evaluate_for_regression(y_test, y_pred):
    # for regression
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import r2_score

    print("mean_absolute_error: ", mean_absolute_error(y_test, y_pred))
    print("mean_squared_error : ", mean_squared_error(y_test, y_pred))
    print("mean_squared_log_error: ", mean_squared_log_error(y_test, y_pred))
    print("r2_score: ", r2_score(y_test, y_pred))

def svr_model(X_train, X_test, y_train, y_test, kernel):

    if kernel == 'linear':
        gsc = GridSearchCV(
            estimator=SVR(kernel='linear'),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_

        print('{} best params : {}'.format('linear', best_params))
        best_svr = SVR(kernel='linear', C=best_params["C"], epsilon=best_params["epsilon"], coef0=0.1, shrinking=True,
                       tol=0.001, cache_size=200, verbose=False, max_iter=-1)

        # Train the model using the training sets
        best_svr.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svr.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_regression(y_test, y_pred)


    elif kernel == 'sigmoid':
        gsc = GridSearchCV(
            estimator=SVR(kernel='sigmoid'),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'degree': [2, 3, 4],
                'coef0': [0.1, 0.01, 0.001, 0.0001],
                'gamma':['auto']},
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_

        print('{} best params : {}'.format('sigmoid', best_params))
        best_svr = SVR(kernel='sigmoid', C=best_params["C"], epsilon=best_params["epsilon"], coef0=best_params["coef0"],                   degree=best_params["degree"], shrinking=True,
                       tol=0.001, cache_size=200, verbose=False, max_iter=-1, gamma='auto')

        # Train the model using the training sets
        best_svr.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svr.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_regression(y_test, y_pred)

    elif kernel == 'poly':
        gsc = GridSearchCV(
            estimator=SVR(kernel='poly'),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'degree': [2, 3, 4],
                'coef0': [0.1, 0.01, 0.001, 0.0001],
                'gamma':['auto']},
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_

        print('{} best params : {}'.format('poly', best_params))
        best_svr = SVR(kernel='poly', C=best_params["C"], epsilon=best_params["epsilon"], coef0=best_params["coef0"],                   degree=best_params["degree"], shrinking=True,
                       tol=0.001, cache_size=200, verbose=False, max_iter=-1, gamma='auto')

        # Train the model using the training sets
        best_svr.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svr.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_regression(y_test, y_pred)


    elif kernel == 'rbf':

        gsc = GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_
        print('{} best params : {}'.format('rbf', best_params))
        best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                       coef0=0.1, shrinking=True,
                       tol=0.001, cache_size=200, verbose=False, max_iter=-1)

        # Train the model using the training sets
        best_svr.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svr.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_regression(y_test, y_pred)

    elif kernel == 'linear-regression':

        gsc = GridSearchCV(
            estimator=LinearRegression(normalize=True,),
            param_grid={
                'fit_intercept': [True, False]
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_
        print('{} best params : {}'.format('rbf', best_params))
        best_svr = LinearRegression(fit_intercept=best_params["fit_intercept"])

        # Train the model using the training sets
        best_svr.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svr.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_regression(y_test, y_pred)


if __name__ == '__main__':

    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    boston.head()

    X = boston.iloc[:, [0, 12]]
    y = boston.iloc[:, 13]

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test

    # Run
    # print(svr_model(X_train, X_test, y_train, y_test, 'linear'))
    # Run
    # print(svr_model(X_train, X_test, y_train, y_test, 'sigmoid'))
    # Run
    # print(svr_model(X_train, X_test, y_train, y_test, 'rbf'))
    print(svr_model(X_train, X_test, y_train, y_test, 'linear-regression'))