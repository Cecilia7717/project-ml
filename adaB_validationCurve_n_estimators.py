from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import utils
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(30)
np.random.seed(30)
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
X = online_shoppers_purchasing_intention_dataset.data.features 

y = online_shoppers_purchasing_intention_dataset.data.targets 
y = y.values.ravel()
X,y = utils.shuffle(X, y)

X = X[:1000]
y = y[:1000]
X = pd.get_dummies(X, columns=['Month', 'VisitorType'])
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30, stratify=y)
n_estimators_range = range(1, 1001)

train_scores = []
val_scores = []

for n_estimators in n_estimators_range:
    ada_model = AdaBoostClassifier(n_estimators=n_estimators, random_state=30)

    train_score = cross_val_score(ada_model, X_train, y_train, cv=5).mean()
    train_scores.append(train_score)
    print(f"Train:{train_score}")

    ada_model.fit(X_train, y_train)
    val_score = ada_model.score(X_test, y_test)
    # val_score = accuracy_score(y_test, ada_model.predict(X_test))
    val_scores.append(val_score)
    print(f"Validation:{val_scores}")

plt.plot(n_estimators_range, train_scores, label='Train')
plt.plot(n_estimators_range, val_scores, label='Validation')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Validation Curve for AdaBoost')
plt.legend()
plt.savefig("Validation Curve for AdaBoost (n_estimators_1).png")
plt.show()