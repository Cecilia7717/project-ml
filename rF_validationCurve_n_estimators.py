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
import random

random.seed(26)
np.random.seed(26)

# fetch dataset 
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
X = online_shoppers_purchasing_intention_dataset.data.features 

y = online_shoppers_purchasing_intention_dataset.data.targets 
y = y.values.ravel()
X,y = utils.shuffle(X, y)

X = X[:1000]
y = y[:1000]
X = pd.get_dummies(X, columns=['Month', 'VisitorType'])
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
n_estimators_range = range(1, 251)

train_scores = []
val_scores = []

for n_estimators in n_estimators_range:
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    train_score = cross_val_score(rf_model, X_train, y_train, cv=5).mean()
    train_scores.append(train_score)
    print(f"Train:{train_score}")

    rf_model.fit(X_train, y_train)
    valid_score = rf_model.score(X_test, y_test)
    val_scores.append(valid_score)
    print(f"Validation:{valid_score}")

plt.plot(n_estimators_range, train_scores, label='Train')
plt.plot(n_estimators_range, val_scores, label='Validation')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Random Forest')
plt.legend()

plt.savefig("Validation Curve for Random Forest (n_estimators).png")
plt.show()