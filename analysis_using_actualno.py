import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_unemployment_data.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Convert 'Date' column to datetime
# df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y", errors='coerce')


# Convert percentage columns to numeric
if df['Estimated Unemployment Rate (%)'].dtype == 'object':  
    df['Estimated Unemployment Rate (%)'] = df['Estimated Unemployment Rate (%)'].str.rstrip('%').astype(float)

# Convert percentage columns to numeric (handle empty values)
df['Estimated Labour Participation Rate (%)'] = (
    df['Estimated Labour Participation Rate (%)']
    .astype(str)
    .str.strip()  # Remove spaces
    .str.rstrip('%')  # Remove '%' symbol
    .replace('', np.nan)  # Replace empty strings with NaN
    .astype(float)  # Convert to float
)


# Calculate Actual Unemployed
df['Actual Unemployed'] = (df['Estimated Employed'] * df['Estimated Unemployment Rate (%)']) / (100 - df['Estimated Unemployment Rate (%)'])

# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Display basic statistics
print("\nBasic statistics:\n", df.describe())

# Display data types of each column
print("\nData types of each column:\n", df.dtypes)

# Correlation matrix heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Actual Unemployed Over Time
plt.figure(figsize=(8, 5))
sns.lineplot(x=df['Date'], y=df['Actual Unemployed'])
plt.xticks(rotation=45)
plt.title("Actual Unemployed Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Unemployed")
plt.show()

# Actual Unemployed by Region
plt.figure(figsize=(12, 6))
sns.barplot(x=df['Region'], y=df['Actual Unemployed'])
plt.xticks(rotation=90)
plt.title("Actual Unemployed by Region")
plt.xlabel("Region")
plt.ylabel("Number of Unemployed")
plt.show()

# Labour Participation Rate by Region
plt.figure(figsize=(12, 6))
sns.barplot(x=df['Region'], y=df['Estimated Labour Participation Rate (%)'])
plt.xticks(rotation=90)
plt.title("Labour Participation Rate by Region")
plt.xlabel("Region")
plt.ylabel("Labour Participation Rate (%)")
plt.show()

# Histogram of Actual Unemployed
plt.figure(figsize=(8, 5))
sns.histplot(df['Actual Unemployed'], bins=20, kde=True, color='Magenta')
plt.title("Distribution of Actual Unemployed")
plt.xlabel("Number of Unemployed")
plt.ylabel("Frequency")
plt.show()

# Boxplot of Actual Unemployed by Region
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Region'], y=df['Actual Unemployed'], color='purple')
plt.xticks(rotation=90)
plt.title("Actual Unemployed Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Number of Unemployed")
plt.show()

# Export cleaned data with Actual Unemployed
df.to_csv("cleaned_unemployment_data.csv", index=False)

