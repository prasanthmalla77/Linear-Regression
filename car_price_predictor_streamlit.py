import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load the dataset (assuming you have a CSV file with car data)
data = pd.read_csv('car_data.csv')  # Replace 'car_data.csv' with your dataset file path

# Create a linear regression model
model = LinearRegression()

# Fit the model to the entire dataset
X = data[['Year', 'Present_Price', 'Kms_Driven', 'Owner']]
y = data['Selling_Price']
model.fit(X, y)

# Streamlit UI
st.title("Car Price Prediction")

# Input fields for user to enter car details
year = st.slider("Select Year:", min_value=2000, max_value=2022, value=2020)
present_price = st.number_input("Present Price (in lakhs):", value=10.0)
kms_driven = st.number_input("Kilometers Driven:", value=50000)
owner = st.slider("Number of Owners:", min_value=0, max_value=3, value=0)

# Predict the selling price
predicted_price = model.predict([[year, present_price, kms_driven, owner]])
st.subheader("Predicted Selling Price:")
st.write(f"The predicted selling price for this car is ₹{predicted_price[0]:.2f} lakhs.")
