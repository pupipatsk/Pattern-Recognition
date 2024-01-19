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