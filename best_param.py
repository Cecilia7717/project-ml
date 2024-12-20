from ucimlrepo import fetch_ucirepo 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

def best_param_prediction(X_train, y_train, X_test, y_test, clf, params):

    # Perform GridSearchCV on the training set
    gridSearch_clf = GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring='accuracy')
    gridSearch_clf.fit(X_train, y_train)

    # Evaluate on the test set using the best parameters 
    best_model = gridSearch_clf.best_estimator_
    y_pred = best_model.predict(X_test)
    feature_importance = best_model.feature_importances_
    print(feature_importance)
    # Evaluate the Best Model
    accuracy = accuracy_score(y_test, y_pred)

    # Best parameters from the grid search
    best_params = gridSearch_clf.best_params_
    print("Best Parameters: ", best_params)
    
    return accuracy, best_params



def main():
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
    X = online_shoppers_purchasing_intention_dataset.data.features 
    y = online_shoppers_purchasing_intention_dataset.data.targets 
    y = y.values.ravel()
    X,y = utils.shuffle(X, y)

    X = X[:1000]
    y = y[:1000]
    X = pd.get_dummies(X, columns=['Month', 'VisitorType'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Define Random Forest classifier and parameter grid
    rf_clf = RandomForestClassifier(random_state=42)
    rf_parameters = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }

    # RF best_param prediction with cv=5
    rf_accuracy, rf_best_params = best_param_prediction(X_train, y_train, X_test, y_test, rf_clf, rf_parameters)
    print(rf_accuracy)

    # Define Adaboost classifier and parameter grid
    ada_clf = AdaBoostClassifier(random_state=42)
    ada_parameters = {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }

    ada_accuracy, ada_best_params = best_param_prediction(X_train, y_train, X_test, y_test, ada_clf, ada_parameters)
    print(ada_accuracy)
    if ada_accuracy > rf_accuracy:
        # Adaboost has better performance with shopper dataset
        pass






if __name__ == "__main__":
    main()


