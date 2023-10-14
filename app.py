# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset (assuming you have a CSV file with car data)
data = pd.read_csv('car_data.csv')  # Replace 'your_car_data.csv' with your dataset file path

# Assuming 'Year', 'Present_Price', 'Kms_Driven', and 'Owner' are important features for prediction
X = data[['Year', 'Present_Price', 'Kms_Driven', 'Owner']]
y = data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# You can use this model to make predictions for new data
new_data = pd.DataFrame({'Year': [2020], 'Present_Price': [10.0], 'Kms_Driven': [50000], 'Owner': [0]})
predicted_price = model.predict(new_data)
print("Predicted Selling Price for New Car Data:", predicted_price[0])
