#datashet : The MNIST database of handwritten digits

from sklearn.datasets import fetch_openml

X, y = fetch_openml ('mnist_784', data_home='./dataset/mnist', return_X_y=True)
X.shape 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

pos = 1 
for data in X[:8]:
    plt.subplot(1, 8, pos)
    plt.imshow(data.reshape((28, 28)), 
    cmap = cm.Greys_r)
    plt.axis('off')
    pos += 1

    plt.show()

y[:8]

#X_train = X[:6000]
#y_train = y[:6000]
#X_test = X[6000:]
#y_test = y[6000:]

X_train = X[:1000]
y_train = y[:1000]
X_test = X[69000:]
y_test = y[69000:]

#clasification dengan SVC (Support Vector Classifer)
from sklearn.svm import SVC
model = SVC(random_state=0)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#Hyperparameter Tuning dengan GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {
    'kernel' : ['rbf', 'poly', 'sigmoid'],
    'C' : [0.5, 1, 10, 100],
    'gamma' : ['scale', 1, 0.1, 0.01, 0.001]
}

grid_search = GridSearchCV(estimator=SVC(random_state=0),
param_grid = parameters,
n_jobs=6,
verbose=1,
scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'Best Score: {grid_search.best_score_}')

best_params = grid_search.best_estimator_.get_params()
print(f'Best Parameters:')
for param in parameters :
    print(f'\t{param}: {best_params[param]}')

#Predict & Evaluate
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
