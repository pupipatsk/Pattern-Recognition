# Logistic Regression classifier using Gradient descent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def GradientDescent(theta, learning_rate, num_iterations, X, y):
    for i in range(num_iterations):
        y_linear = np.dot(X, theta)
        h = sigmoid(y_linear)
        gradient = np.dot((y - h), X)
        # update rule
        theta = theta + learning_rate * gradient
    return theta # best theta

def predict(X, theta):
    y_linear = np.dot(X, theta)
    h = sigmoid(y_linear)
    return np.where(h >= 0.5, 1, 0)

# Main function
X = data
y = np.array(train['Survived'].values)
m, n = data.shape # m = number of sample, n = number of features

# Initialize theta
theta = np.zeros(n)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Gradient Descent # to minimize Cost Function
theta = GradientDescent(theta, learning_rate, num_iterations, X, y) # best theta

print(f'Best theta: {theta}')

# Confusion Matrix
from sklearn.metrics import accuracy_score, confusion_matrix

y_test_cfm = np.array(train['Survived'].values)
y_pred_cfm = predict(X, theta)

# Determine model accuracy and goodness of fit
accuracy_value = accuracy_score(y_test_cfm, y_pred_cfm, normalize=True)
conf_mat = confusion_matrix(y_test_cfm, y_pred_cfm)

print("The accuracy of the model is:", accuracy_value)
print("Confusion Matrix:\n", conf_mat)

# # save prediction to .csv
# prediction_data = pd.DataFrame({ 
#     "PassengerId": test["PassengerId"],
#     "Survived": y_pred[0]
# })
# prediction_data.to_csv("submission.csv", index=False)

# ---------------------------------------------------------------------- #

# Linear Regression classifier using Gradient descent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

def GradientDescent(theta, learning_rate, num_iterations, X, y):
    cost = []
    for i in range(num_iterations):
        y_linear = np.dot(X, theta[1:]) + theta[0]
        # h = sigmoid(y_linear)
        gradient = np.dot((y - y_linear), X)
        # update rule
        # theta = theta + learning_rate * gradient
        theta[0] = theta[0] + learning_rate * (y - y_linear).sum()
        theta[1:] = theta[1:] + learning_rate * gradient
        
        cost.append(CostFunction (theta, X, y))
    return theta, cost # best theta

def predict(X, theta):
    y_linear = np.dot(X, theta[1:]) + theta[0]
    # h = sigmoid(y_linear)
    return np.where(y_linear >= 0.5, 1, 0)

def CostFunction (theta, X, y):
    y_linear = np.dot(X, theta[1:]) + theta[0]
    errors = y - y_linear
    cost = np.sum(errors**2) / m
    return cost

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

# Main function
X = normalize(data) # "Pclass","Sex","Age","Embarked"
y = np.array(train['Survived'].values)
m, n = data.shape # m = number of sample, n = number of features

# Initialize theta
theta = np.zeros(1+n)

# Set hyperparameters
learning_rate = 0.001
num_iterations = 100

# Gradient Descent # to minimize Cost Function
theta, cost = GradientDescent(theta, learning_rate, num_iterations, X, y) # best theta

print(f'Best theta: {theta}')
print(f'Cost: {cost}\n')

# Plot the cost function
plt.plot(range(1, num_iterations+1), cost, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()
# ---------------------------------------------------------------------- #

# Linear Regression classifier using Matrix Inversion

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict(X, theta):
    y_linear = np.dot(X, theta[1:]) + theta[0]
    # h = sigmoid(y_linear)
    return np.where(y_linear >= 0.5, 1, 0)

def CostFunction (theta, X, y):
    y_linear = np.dot(X, theta)
    errors = y - y_linear
    cost = np.sum(errors**2) / m
    return cost

# Main function
X = data # "Pclass","Sex","Age","Embarked"
X = np.insert(X, 0, 1, axis=1) # "1(for theta0)", Pclass","Sex","Age","Embarked"
y = np.array(train['Survived'].values)
m, n = data.shape # m = number of sample, n = number of features

# Initialize theta
theta = np.zeros(1+n)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Calculate best theta
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Calculate the cost
cost = CostFunction(theta, X, y)

print(f'Best theta: {theta}')
print(f'Cost: {cost}')