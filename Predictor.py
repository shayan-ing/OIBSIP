# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load the Dataset
df = pd.read_csv('car data.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Data Preprocessing
# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Drop irrelevant columns (e.g., 'Car_Name')
df = df.drop('Car_Name', axis=1)

# Step 4: Feature Engineering
# Add new features: Age of the Car and Price Depreciation
df['Age'] = 2025 - df['Year']  #  the current year is 2025

# Display the updated dataset
print("\nDataset after adding new features:")
print(df.head())

# Separate features (X) and target (y)
X = df.drop('Selling_Price(In Lakhs)', axis=1)  # Features
y = df['Selling_Price(In Lakhs)']  # Target variable

# Step 5: Exploratory Data Analysis (EDA)
# 5.1: Summary Statistics
print("\nSummary Statistics for Numerical Columns:")
print(df.describe())

# 5.2: Distribution of Numerical Features
plt.figure(figsize=(12, 6))
for i, col in enumerate(['Year', 'Present_Price(In Lakhs)', 'Driven_kms', 'Owner', 'Age']):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# 5.3: Distribution of Categorical Features
plt.figure(figsize=(12, 6))
for i, col in enumerate(['Fuel_Type', 'Selling_type', 'Transmission']):
    plt.subplot(1, 3, i + 1)
    sns.countplot(data=df, x=col)
    plt.title(f'Count of {col}')
plt.tight_layout()
plt.show()

# 5.4: Correlation Heatmap (only for numerical columns)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Numerical Features')
plt.show()

# Step 6: Feature Engineering
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Step 7: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Machine Learning Model
# Use Random Forest Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Step 9: Evaluate the Model
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 10: Take User Input and Predict
def predict_price():
    print("\nEnter the following details to predict the car's selling price:")
    
    # Take user input
    year = int(input("Year of the car: "))
    present_price = float(input("Present Price (In Lakhs): "))
    driven_kms = int(input("Driven Kilometers: "))
    fuel_type = input("Fuel Type (Petrol/Diesel/CNG): ")
    selling_type = input("Selling Type (Dealer/Individual): ")
    transmission = input("Transmission (Manual/Automatic): ")
    owner = int(input("Number of Owners (0, 1, 2, 3): "))
    
    # Calculate Age and Price Depreciation
    age = 2025 - year  # Assuming the current year is 2025

    
    # Create a DataFrame from user input
    user_input = pd.DataFrame({
        'Year': [year],
        'Present_Price(In Lakhs)': [present_price],
        'Driven_kms': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission],
        'Owner': [owner],
        'Age': [age],
    })
    
    # Predict the selling price
    predicted_price = model.predict(user_input)
    print(f"\nPredicted Selling Price (In Lakhs): {predicted_price[0]:.2f}")

# Call the function to take user input and predict
predict_price()
import joblib

# Save the trained model
joblib.dump(model, 'car_price_predictor.pkl')
print("\nModel saved as 'car_price_predictor.pkl'")
