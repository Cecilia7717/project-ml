from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import utils
from ucimlrepo import fetch_ucirepo
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

max_depth_range = range(1, 51)

# Initialize empty lists to store validation scores
train_scores = []
val_scores = []

# Iterate over different max_depth values
for max_depth in max_depth_range:
    # Create a Random Forest model with the current max_depth
    rf_model = RandomForestClassifier(max_depth=max_depth, random_state=42)

    # Cross-validation on training data
    train_score = cross_val_score(rf_model, X_train, y_train, cv=5).mean()
    train_scores.append(train_score)

    # Evaluate on validation set
    rf_model.fit(X_train, y_train)
    val_score = rf_model.score(X_test, y_test)
    val_scores.append(val_score)

# Plot the validation curve
plt.plot(max_depth_range, train_scores, label='Train')
plt.plot(max_depth_range, val_scores, label='Validation', linestyle='--')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Random Forest')
plt.xticks(range(min(max_depth_range), max(max_depth_range) + 1, 5))  # Adjust range if needed

# Rotate x-axis labels for better readability
plt.xticks(rotation=45) 
plt.legend()
plt.savefig("Validation Curve for Random Forest (max_depth).png")
plt.show()