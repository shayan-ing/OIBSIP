# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Step 2: Load the Dataset
df = pd.read_csv('car data.csv')

# Step 3: Data Preprocessing
# Drop irrelevant columns (e.g., 'Car_Name')
df = df.drop('Car_Name', axis=1)

# Separate features (X) and target (y)
X = df.drop('Selling_Price(In Lakhs)', axis=1)  # Features
y = df['Selling_Price(In Lakhs)']  # Target variable

# Step 4: Feature Engineering
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Machine Learning Model
# Use Random Forest Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Step 7: Take User Input and Predict
def predict_price():
    print("Enter the following details to predict the car's selling price:")
    
    # Take user input
    year = int(input("Year of the car: "))
    present_price = float(input("Present Price (In Lakhs): "))
    driven_kms = int(input("Driven Kilometers: "))
    fuel_type = input("Fuel Type (Petrol/Diesel/CNG): ")
    selling_type = input("Selling Type (Dealer/Individual): ")
    transmission = input("Transmission (Manual/Automatic): ")
    owner = int(input("Number of Owners (0, 1, 2, 3): "))
    
    # Create a DataFrame from user input
    user_input = pd.DataFrame({
        'Year': [year],
        'Present_Price(In Lakhs)': [present_price],
        'Driven_kms': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })
    
    # Predict the selling price
    predicted_price = model.predict(user_input)
    print(f"\nPredicted Selling Price (In Lakhs): {predicted_price[0]:.2f}")

# Call the function to take user input and predict
predict_price()