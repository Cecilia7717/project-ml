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
n_estimators_range = range(1, 151)
# learning_rate_range = np.linspace(0.00001, 1.0, 100)  
learning_rate_range = np.linspace(0.00001, 0.1, 100)  
train_scores = []
val_scores = []

for learning_rate in learning_rate_range:
    ada_model = AdaBoostClassifier(learning_rate = learning_rate, random_state=30)

    train_score = cross_val_score(ada_model, X_train, y_train, cv=5).mean()
    train_scores.append(train_score)

    ada_model.fit(X_train, y_train)
    val_score = ada_model.score(X_test, y_test)
    val_scores.append(val_score)

plt.plot(learning_rate_range, train_scores, label='Train')
plt.plot(learning_rate_range, val_scores, label='Validation', linestyle='--')

# plt.xticks(5)
plt.xticks(rotation=45) 
import numpy as np
# plt.xticks(np.arange(min(learning_rate_range), max(learning_rate_range), 0.2))
plt.xticks(np.arange(min(learning_rate_range), max(learning_rate_range), 0.02))
# plt.ylim(0.82,0.95)
plt.xlabel('learning rate')
plt.ylabel('Accuracy')
plt.grid()
plt.title('Validation Curve for AdaBoost')
plt.legend()
plt.savefig("Validation Curve for AdaBoost (learning_rate).png")
plt.show()