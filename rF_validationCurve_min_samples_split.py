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

random.seed(30)
np.random.seed(30)
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
 

min_samples_split_range = range(2, 41)

train_scores = []
val_scores = []

for min_samples_split in min_samples_split_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30, stratify=y)
    
    rf_model = RandomForestClassifier(min_samples_split=min_samples_split, random_state=30)

    train_score = cross_val_score(rf_model, X_train, y_train, cv=5).mean()
    train_scores.append(train_score)

    rf_model.fit(X_train, y_train)
    val_score = rf_model.score(X_test, y_test)
    val_scores.append(val_score)

plt.plot(min_samples_split_range, train_scores, label='Train')
plt.plot(min_samples_split_range, val_scores, label='Validation', linestyle='--')
plt.xlabel('min_samples_split')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Random Forest')
plt.xticks(range(min(min_samples_split_range), max(min_samples_split_range), 5)) 
# plt.xticks(rotation=45) 
plt.legend()
plt.savefig("Validation Curve for Random Forest (min_samples_split).png")
plt.show()