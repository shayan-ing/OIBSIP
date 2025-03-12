import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Unemployment_data.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Convert percentage columns to numeric
if df['Estimated Unemployment Rate (%)'].dtype == 'object':  
    df['Estimated Unemployment Rate (%)'] = df['Estimated Unemployment Rate (%)'].str.rstrip('%').astype(float)

if df['Estimated Labour Participation Rate (%)'].dtype == 'object':  
    df['Estimated Labour Participation Rate (%)'] = df['Estimated Labour Participation Rate (%)'].str.rstrip('%').astype(float)

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

# Unemployment Rate Over Time
plt.figure(figsize=(8, 5))
sns.lineplot(x=df['Date'], y=df['Estimated Unemployment Rate (%)'])
plt.xticks(rotation=45)
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# Unemployment Rate by Region
plt.figure(figsize=(12, 6))
sns.barplot(x=df['Region'], y=df['Estimated Unemployment Rate (%)'])
plt.xticks(rotation=90)
plt.title("Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# Labour Participation Rate by Region
plt.figure(figsize=(12, 6))
sns.barplot(x=df['Region'], y=df['Estimated Labour Participation Rate (%)'])
plt.xticks(rotation=90)
plt.title("Labour Participation Rate by Region")
plt.xlabel("Region")
plt.ylabel("Labour Participation Rate (%)")
plt.show()

# Histogram of Unemployment Rate
plt.figure(figsize=(8, 5))
sns.histplot(df['Estimated Unemployment Rate (%)'], bins=20, kde=True, color='Magenta')
plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.show()

# Boxplot of Unemployment Rate by Region
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Region'], y=df['Estimated Unemployment Rate (%)'], color='purple')
plt.xticks(rotation=90)
plt.title("Unemployment Rate Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# Export cleaned data
df.to_csv("cleaned_unemployment_data.csv", index=False)
