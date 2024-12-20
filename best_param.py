from ucimlrepo import fetch_ucirepo 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, random

from sklearn import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.ensemble._weight_boosting")

random.seed(30)
np.random.seed(30)

def best_param_prediction(X_train, y_train, X_test, y_test, clf, params):

    # Perform GridSearchCV on the training set
    gridSearch_clf = GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring='accuracy')
    gridSearch_clf.fit(X_train, y_train)

    # Evaluate on the test set using the best parameters 
    best_model = gridSearch_clf.best_estimator_
    y_pred = best_model.predict(X_test)
    feature_importance = best_model.feature_importances_

    # Evaluate the Best Model
    accuracy = accuracy_score(y_test, y_pred)

    # Best parameters from the grid search
    best_params = gridSearch_clf.best_params_
    
    return accuracy, best_params, feature_importance


def important_F(X, rf_F, ada_F):
    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'RF Importance': rf_F,
        'AdaBoost Importance': ada_F
    })

    feature_importances = feature_importances.sort_values(
        by=['RF Importance', 'AdaBoost Importance'],  
        ascending=[False, False] 
    )

    # Define bar positions
    features = feature_importances['Feature']
    rf_importance = feature_importances['RF Importance']
    ada_importance = feature_importances['AdaBoost Importance']

    # Bar positions and width
    x = range(len(features))  # The label locations
    width = 0.40  # The width of the bars

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Add bars for each group
    bars1 = ax.bar([p - width / 2 for p in x], rf_importance, width, label='RF Importance')
    bars2 = ax.bar([p + width / 2 for p in x], ada_importance, width, label='AdaBoost Importance')

    # Add labels, title, and legend
    ax.set_xlabel('Features Name')
    ax.set_ylabel('Importance')
    ax.set_title('Online Shoppers Purchasing Intention Dataset - Feature Importance From RF Classifier and AdaBoost Classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    # Rotate x-axis labels to avoid overlap
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    plt.savefig("Important_Features_side-by-side.png")

    table_markdown = feature_importances.to_markdown(index=False)
    with open('importance_table.md', 'w') as f:
        f.write(table_markdown)

def evaluate_model(name, y_test, y_pred):
    """Compute metrics for a given model."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def main():
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
    X = online_shoppers_purchasing_intention_dataset.data.features 
    y = online_shoppers_purchasing_intention_dataset.data.targets 
    y = y.values.ravel()
    X,y = utils.shuffle(X, y)

    # X = X[:1000]
    # y = y[:1000]
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
    rf_accuracy, rf_best_params, rf_importantf = best_param_prediction(X_train, y_train, X_test, y_test, rf_clf, rf_parameters)
    # print("random forest accuracy: ", rf_accuracy)
    print("RF - best parameters: ")
    print(rf_best_params)

    # Define Adaboost classifier and parameter grid
    ada_clf = AdaBoostClassifier(random_state=42)
    ada_parameters = {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }

    # AdaBoost best_param prediction with cv=5
    ada_accuracy, ada_best_params, ada_importantf = best_param_prediction(X_train, y_train, X_test, y_test, ada_clf, ada_parameters)
    # print("Adaboost accuracy: ", ada_accuracy)
    print("Adaboost - best parameters: ")
    print(ada_best_params)

    # generate results with important features from the two models
    important_F(X, rf_importantf, ada_importantf)

    # Evaluate both models and generate a comparison table
    rf_best_model = RandomForestClassifier(random_state=42, **rf_best_params)
    rf_pred = rf_best_model.fit(X_train, y_train).predict(X_test)

    ada_best_model = AdaBoostClassifier(random_state=42, **ada_best_params)
    ada_pred = ada_best_model.fit(X_train, y_train).predict(X_test)

    metrics = [
        evaluate_model("Random Forest", y_test, rf_pred),
        evaluate_model("AdaBoost", y_test, ada_pred)
    ]

    metrics_df = pd.DataFrame(metrics)

    # Save and display metrics table
    print(metrics_df.to_markdown(index=False))
    with open("model_comparison_table.md", "w") as f:
        f.write(metrics_df.to_markdown(index=False))
    

    # confusion matrix with the algorithm that has higher accuracy
    if rf_accuracy > ada_accuracy:
        best_model = rf_best_model
        y_pred = rf_pred
        
    else:
        best_model = ada_best_model
        y_pred = ada_pred

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    

if __name__ == "__main__":
    main()


