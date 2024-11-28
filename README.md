# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: vijay k
RegisterNumber: 24901153
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear regression function
def linear_regression(X, y, num_iters=1000, learning_rate=0.01):
    theta = np.zeros((X.shape[1], 1))  # Initialize theta as a column vector of zeros
    for _ in range(num_iters):
        predictions = X.dot(theta)  # Predicted values
        errors = predictions - y  # Prediction errors
        gradient = (1 / len(X)) * X.T.dot(errors)  # Compute gradient
        theta -= learning_rate * gradient  # Update theta
    return theta

# Load the dataset
data = pd.read_csv('/content/50_Startups.csv')

# Prepare the features and target variable
X = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values.reshape(-1, 1)  # The last column

# Convert data to float
X = X.astype(float)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Add a column of ones for the bias term (intercept)
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# Train the model
theta = linear_regression(X_scaled, y_scaled)

# Predict for a new data point
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(1, -1)
new_scaled = scaler_X.transform(new_data)
new_scaled = np.hstack((np.ones((new_scaled.shape[0], 1)), new_scaled))  # Add bias term

prediction_scaled = new_scaled.dot(theta)
prediction = scaler_y.inverse_transform(prediction_scaled)

print(f"Predicted value: {prediction[0][0]}")
 
*/
```

## Output:
![image](https://github.com/user-attachments/assets/82f4afb5-b0f5-422d-b18c-13cce08060ec)
![image](https://github.com/user-attachments/assets/83b21514-886c-4325-966b-553c18e5ed01)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
