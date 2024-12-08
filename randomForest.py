from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
# fetch dataset 
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
# data (as pandas dataframes) 
X = online_shoppers_purchasing_intention_dataset.data.features 

y = online_shoppers_purchasing_intention_dataset.data.targets 
y = y.values.ravel()
X,y = utils.shuffle(X, y)

X = X[:1000]
y = y[:1000]
X = pd.get_dummies(X, columns=['Month', 'VisitorType'])
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Define a parameter grid with additional parameters
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees in the forest
    'max_depth': [None, 10, 20],           # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split an internal node
    'max_features': ['sqrt', 'log2', None], # Number of features to consider when looking for the best split
    'criterion': ['gini', 'entropy'],      # Function to measure the quality of a split
}

# Initialize GridSearch
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=1)

# Fit to the training data
grid_search.fit(X_train, y_train)

# Display best parameters and corresponding score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Extract the best parameters from the GridSearchCV
best_params = grid_search.best_params_

# Initialize the Random Forest model with the best parameters
best_rf_model = RandomForestClassifier(random_state=42, 
                                       n_estimators=best_params['n_estimators'],
                                       max_depth=best_params['max_depth'],
                                       min_samples_split=best_params['min_samples_split'],
                                       max_features=best_params['max_features'],
                                       criterion=best_params['criterion'])

# Fit the model on the training data
best_rf_model.fit(X_train, y_train)

# Extract feature importances
importances = best_rf_model.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("table:", feature_importances)

# Display the top features
top_n = 5  # Adjust this to show more or fewer features
top_features = feature_importances.head(top_n)
print("Top Features:\n", top_features)

# Plot the top features
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Feature Importances")
plt.gca().invert_yaxis()  # To display the most important features at the top
plt.savefig("Top 5 Important Features.png")
plt.show()
