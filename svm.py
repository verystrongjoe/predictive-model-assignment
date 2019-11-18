from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate
import math
import numpy as np


def evaluate_for_classfication(y_test, y_pred):
    # for classification
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import roc_auc_score

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("f1_score: ", f1_score(y_test, y_pred))
    print("recall_score: ", recall_score(y_test, y_pred))
    print("precision_score: ", precision_score(y_test, y_pred))
    print("roc_auc_score: ", roc_auc_score(y_test, y_pred))


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

cancer = datasets.load_breast_cancer()

print(cancer.data.shape)
print(cancer.data[0:5])
print(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test


"""
Tuning Hyperparameters
Kernel: The main function of the kernel is to transform the given dataset input data into the required form. There are various types of functions such as linear, polynomial, and radial basis function (RBF). Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.
Regularization: Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty 
parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
Gamma: A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.
"""
def svm_model(X_train, X_test, y_train, y_test, kernel):
    print('------------------------*****{}******----------------------'.format(kernel))
    if kernel == 'linear':
        gsc = GridSearchCV(
            estimator=svm.SVC(kernel=kernel),
            param_grid={
                'C': [0.1, 1, 100, 1000]},
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_

        # Create best svm Classifier
        best_svm = svm.SVC(kernel=kernel, C=best_params["C"])

        print('{} best params : {}'.format(kernel, best_params))

        # Train the model using the training sets
        best_svm.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svm.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_classfication(y_test, y_pred)

    elif kernel == 'poly':

        print('test1')
        gsc = GridSearchCV(
            estimator=svm.SVC(kernel=kernel),
            param_grid={
                'C': [100],
                'gamma': [1, 5]},
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        print('test2')

        # gsc = GridSearchCV(
        #     estimator=svm.SVC(kernel=kernel),
        #     param_grid={
        #         'C': [0.1, 1, 100],
        #         'gamma': [0.0001, 0.005, 0.1, 1, 5]},
        #     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        print('test3')

        grid_result = gsc.fit(X_train, y_train)
        print('test4')
        best_params = grid_result.best_params_
        print('{} best params : {}'.format(kernel, best_params))

        best_svm = svm.SVC(kernel=kernel, C=best_params["C"], gamma=best_params["gamma"], coef0=best_params["coef0"])

        # Train the model using the training sets
        best_svm.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svm.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_classfication(y_test, y_pred)

    elif kernel == 'rbf':
        gsc = GridSearchCV(
            estimator=svm.SVC(kernel=kernel),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'gamma': [0.0001, 0.005, 0.1, 1, 5]},
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        print('{} best params : {}'.format(kernel, best_params))

        best_svm = svm.SVC(kernel=kernel, C=best_params["C"], gamma=best_params["gamma"])

        # Train the model using the training sets
        best_svm.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svm.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_classfication(y_test, y_pred)


    elif kernel == 'sigmoid':
        gsc = GridSearchCV(
            estimator=svm.SVC(kernel=kernel),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'gamma': [0.0001, 0.005, 0.1, 1, 5],
                'coef0': [0.1, 0.01, 0.0001]},
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        print('{} best params : {}'.format(kernel, best_params))
        best_svm = svm.SVC(kernel=kernel, C=best_params["C"], gamma=best_params["gamma"], coef0=best_params["coef0"])

        # Train the model using the training sets
        best_svm.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = best_svm.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        evaluate_for_classfication(y_test, y_pred)



svm_model(X_train, X_test, y_train, y_test, 'linear')
# svm_model(X_train, X_test, y_train, y_test, 'poly')
svm_model(X_train, X_test, y_train, y_test, 'rbf')
svm_model(X_train, X_test, y_train, y_test, 'sigmoid')
