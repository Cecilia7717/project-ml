from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import utils
from sklearn.metrics import classification_report, accuracy_score
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
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
n_estimators_range = range(1, 151)
learning_rate_range = np.linspace(0.1, 1.0, 20)  # Range for learning rate

# Initialize empty lists to store validation scores
train_scores = []
val_scores = []

# Iterate over different max_depth values
for learning_rate in learning_rate_range:
    # Create a Random Forest model with the current max_depth
    ada_model = AdaBoostClassifier(learning_rate = learning_rate, random_state=42)

    # Cross-validation on training data
    train_score = cross_val_score(ada_model, X_train, y_train, cv=5).mean()
    train_scores.append(train_score)

    # Evaluate on validation set
    ada_model.fit(X_train, y_train)
    val_score = ada_model.score(X_test, y_test)
    val_scores.append(val_score)

# Plot the validation curve
plt.plot(learning_rate_range, train_scores, label='Train')
plt.plot(learning_rate_range, val_scores, label='Validation', linestyle='--')
plt.xlabel('min_samples_split')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Random Forest')
plt.xticks(learning_rate_range)  # Adjust range if needed

# Rotate x-axis labels for better readability
plt.xticks(rotation=45) 

plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Validation Curve for AdaBoost')
plt.savefig("Validation Curve for AdaBoost (learning_rate).png")
plt.legend()
plt.show()