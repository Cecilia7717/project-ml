# datasets
from sklearn.datasets import load_breast_cancer, fetch_20newsgroups_vectorized, fetch_openml

# python packages
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, utils
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='run pipeline')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='name of the dataset used')
    args = parser.parse_args()
    return args

def run_VC(clf, X, y, p_name, p_range):
    train_scores, test_scores = validation_curve(clf, X, y, param_name=p_name, param_range=p_range, cv=3)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    train_mean = np.round(train_mean, 4)
    test_mean = np.round(test_mean, 4) 

    df = pd.DataFrame({
        p_name:p_range, 
        'Train Accuracy':train_mean, 
        'Test Accuracy':test_mean
    })
    print(df.to_string(index=False))
    df.to_csv(f'{p_name}.csv', index=False)
    return df

def generate_graph(df, p_name):
    plt.figure()
    plt.plot(df[p_name], df['Train Accuracy'], label= 'Train Accuracy', marker='o', color='blue')
    plt.plot(df[p_name], df['Test Accuracy'], label = 'Test Accuracy', marker='*', color='red')
    if p_name == 'gamma':
        plt.xscale('log')
    plt.xlabel(f'{p_name}')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(f"{p_name}.png", format='png')


def main():
    args = parse_args()
    if args.dataset == "cancer":
        data = load_breast_cancer()
        X = data['data']
        y = data['target']
    elif args.dataset == "news":
        data = fetch_20newsgroups_vectorized(subset='all')
        X = data['data']
        y = data['target']
        X,y = utils.shuffle(X,y) # shuffle the rows 
        X = X[:1000] # only keep 1000 examples
        y = y[:1000]
    elif args.dataset == "mnist":
        data = fetch_openml('mnist_784', data_home="/Users/ali/Desktop/cs383/HW07")
        X = data['data']
        y = data['target']
        X,y = utils.shuffle(X,y) # shuffle the rows
        X = X[:1000] # only keep 1000 examples
        y = y[:1000]
        X = X/255 # normalize the feature values
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # # KNN
    # knn_clf = KNeighborsClassifier()
    # n_neighbors_values = range(1, 22, 2)
    # run_VC(knn_clf, X, y, 'n_neighbors', n_neighbors_values)

    # Random Forest
    rf_clf = RandomForestClassifier()
    n_estimator_values = range(1, 202, 10)
    df_rf = run_VC(rf_clf, X, y, 'n_estimators', n_estimator_values)
    generate_graph(df_rf, 'n_estimators')

    

    # Support Vector Machine
    svm_clf = svm.SVC()
    gamma_values = np.logspace(-4, 1, num=6)
    df_svm = run_VC(svm_clf, X, y, 'gamma', gamma_values)
    generate_graph(df_svm, 'gamma')

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
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
X = online_shoppers_purchasing_intention_dataset.data.features 

y = online_shoppers_purchasing_intention_dataset.data.targets 
y = y.values.ravel()
X,y = utils.shuffle(X, y)

X = X[:1000]
y = y[:1000]
X = pd.get_dummies(X, columns=['Month', 'VisitorType'])
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
n_estimators_range = range(1, 151)
learning_rate_range = np.linspace(0.00001, 1.0, 100)  

train_scores = []
val_scores = []

train_scores, test_scores = validation_curve(AdaBoostClassifier(), X_train, y_train, param_name="learning_rate", param_range=learning_rate_range, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

train_mean = np.round(train_mean, 4)
test_mean = np.round(test_mean, 4) 

df = pd.DataFrame({
    'learning_rate':learning_rate_range, 
    'Train Accuracy':train_mean, 
    'Test Accuracy':test_mean
})
print(df.to_string(index=False))
p_name = 'learning_rate'
df.to_csv(f'{p_name}.csv', index=False)

plt.figure()
plt.plot(df[p_name], df['Train Accuracy'], label= 'Train Accuracy', marker='o', color='blue')
plt.plot(df[p_name], df['Test Accuracy'], label = 'Test Accuracy', marker='*', color='red')
if p_name == 'gamma':
    plt.xscale('log')
plt.xlabel(f'{p_name}')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy')
plt.legend()
# plt.show()
plt.savefig(f"{p_name}.png", format='png')