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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
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

n_estimators_range = range(1, 201)
criterion_options = ["gini", "entropy", "log_loss"]  # List of splitting criteria

# Initialize empty dictionaries to store validation scores for each criterion
train_scores_dict = {}
val_scores_dict = {}

# Iterate over different criterion options
for criterion in criterion_options:
  # Initialize empty lists for current criterion
  train_scores = []
  val_scores = []

  # Iterate over different n_estimators values
  for n_estimators in n_estimators_range:
    # Create a Random Forest model with the current hyperparameters
    rf_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=42)

    # Cross-validation on training data
    train_score = cross_val_score(rf_model, X_train, y_train, cv=5).mean()
    train_scores.append(train_score)

    # Evaluate on validation set
    rf_model.fit(X_train, y_train)
    val_score = rf_model.score(X_test, y_test)
    val_scores.append(val_score)

  # Store scores for the current criterion
  train_scores_dict[criterion] = train_scores
  val_scores_dict[criterion] = val_scores

# Plot the validation curves for different criteria
for criterion in criterion_options:
  plt.plot(n_estimators_range, train_scores_dict[criterion], label=f'Train ({criterion})')
  plt.plot(n_estimators_range, val_scores_dict[criterion], label=f'Validation ({criterion})', linestyle='--')

plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Random Forest')
plt.legend()
plt.savefig("Validation Curve for Random Forest (criteria).png")
plt.show()