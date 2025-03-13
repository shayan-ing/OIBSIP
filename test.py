import pandas as pd
import joblib

# Load the trained model
model = joblib.load('car_price_predictor.pkl')

# Input data
user_input = pd.DataFrame({
    'Year': [2010],
    'Present_Price(In Lakhs)': [4.50],
    'Driven_kms': [20000],
    'Fuel_Type': ['Petrol'],
    'Selling_type': ['Individual'],
    'Transmission': ['Manual'],
    'Owner': [1],
    'Age': [15]  # 2025 - 2010
})

# Make a prediction
predicted_price = model.predict(user_input)[0]
print(f"Predicted Selling Price (In Lakhs): {predicted_price:.2f}")