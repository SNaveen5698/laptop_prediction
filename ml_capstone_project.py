import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the data
laptop = pd.read_csv("C:\\Users\\yello\\Downloads\\laptop.csv")
df = laptop.copy()

# Display basic information about the dataset
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.describe())
print(df.nunique())
print(df.isnull().sum())
print(df.columns)

# Data preprocessing
df.drop(["Unnamed: 0.1", "Unnamed: 0"], axis=1, inplace=True)
df["Company"].fillna(df["Company"].mode()[0], inplace=True)
df["TypeName"].fillna(df["TypeName"].mode()[0], inplace=True)
df["Inches"].replace({"?": np.nan}, inplace=True)
df["Inches"] = df["Inches"].astype(float)
df["Inches"].fillna(df["Inches"].mean(), inplace=True)
df["ScreenResolution"].fillna(df["ScreenResolution"].mode()[0], inplace=True)
df["Cpu"].fillna(df["Cpu"].mode()[0], inplace=True)
df["Ram"].fillna(df["Ram"].mode()[0], inplace=True)
df["Memory"].fillna(df["Memory"].mode()[0], inplace=True)
df["Gpu"].fillna(df["Gpu"].mode()[0], inplace=True)
df["OpSys"].fillna(df["OpSys"].mode()[0], inplace=True)
df['Weight'] = df['Weight'].str.extract(r'(\d+\.\d+)').astype(float)
df["Weight"].fillna(df["Weight"].mean(), inplace=True)
df["Price"].fillna(df["Price"].mean(), inplace=True)
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Ram'] = df['Ram'].astype('int32')
df = pd.get_dummies(df, columns=['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys'], drop_first=True)

# Ensure all column names are valid and are strings
df.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(',', '').replace(' ', '_') for col in df.columns]

# Splitting the dataset into train and test sets for all companies
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Initialize and train the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2) score: {r2}")

# Plotting actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
