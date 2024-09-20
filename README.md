# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
```
step 1. start the program.
step 2. Load and preprocess the data (define features and target).
step 3. Split the dataset into training and testing sets.
step 4. Scale the features using StandardScaler.
step 5. Train the SGDRegressor model on the training set.
step 6. Evaluate the model on both training and testing sets using MSE or other metrics.
step 7. end the program.
```
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: FRANKLIN RAJ G
RegisterNumber:  212223230058
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the California Housing dataset
data=fetch_california_housing()

# Use the first 3 features as inputs
X = data.data[:,:3] # Features : 'MedInc' ,'House Age','AveRooms'

# Use 'MedHouseVal' and 'AveOccup' as output variables
Y = np.column_stack((data.target,data.data[:,6])) # Targets : 'Med HouseVal',AveOccup'

# Split the data into training and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)

# scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the orginal scale.
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evalulate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:",mse)

# Optionally, print some predictions
print("\nPredictions:\n",Y_pred[:5]) #Print tfirst 5 predictions
*/
```

## Output:
## Mean Squared Error and Predictions:

![ML exp4 out](https://github.com/user-attachments/assets/0e25a9bd-05fe-4cbe-9f17-c6244d3dcbb8)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
