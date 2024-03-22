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
df=laptop
df1=laptop
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.describe())
print(df.nunique())
print(df.isnull().sum())
print(df.columns)
print(df.dtypes)
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
# Splitting the dataset into train and test sets for all companies
x = df.drop("Price", axis=1)
y = df["Price"]
x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(x, y, test_size=0.15, random_state=0)
# Splitting the dataset into train and test sets for all companies
x = df.drop("Price", axis=1)
y = df["Price"]
x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(x, y, test_size=0.15, random_state=0)
scaler = MinMaxScaler()
x_test_scaled = scaler.fit_transform(x_test_all)

# Create a pipeline for all companies
pipeline_all = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', XGBRegressor())
])

# Fit the pipeline on the training data for all companies
pipeline_all.fit(x_train_all, y_train_all)

# Making predictions for all companies
y_pred_all = pipeline_all.predict(x_test_all)

# Evaluate the model for all companies
mae_all = mean_absolute_error(y_test_all, y_pred_all)
mse_all = mean_squared_error(y_test_all, y_pred_all)
r2_all = r2_score(y_test_all, y_pred_all)

print("\nModel Evaluation for All Companies:")
print("Mean Absolute Error (MAE):", mae_all)
print("Mean Squared Error (MSE):", mse_all)
print("R-squared (R2) Score:", r2_all)
# Visualize feature importances for all companies
importances_all = pipeline_all.named_steps['model'].feature_importances_
features_all = x.columns
indices_all = np.argsort(importances_all)[::-1]
top_features_all = 5  # Change this value to show more or fewer top features
plt.figure(figsize=(10, 6))
plt.title("Top {} Feature Importances for All Companies".format(top_features_all))
plt.bar(range(top_features_all), importances_all[indices_all][:top_features_all], align="center")
plt.xticks(range(top_features_all), features_all[indices_all][:top_features_all], rotation=45, ha='right')
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
print("2  Filter lesser-known brands prediction")
lesser_known_brands = ['Acer', 'Asus']
lesser_known_indices = df1[df1['Company'].isin(lesser_known_brands)].index

# Filter scaled features for lesser-known brands
lesser_known_indices = [idx for idx in lesser_known_indices if idx < len(x_test_scaled)]
x_lesser_known_scaled = x_test_scaled[lesser_known_indices]

# Predict for lesser-known brands
y_pred_lesser_known = pipeline_all.predict(x_lesser_known_scaled)

# Print predicted prices
print("Predicted prices for lesser-known brands:")
print(y_pred_lesser_known)

# Calculate evaluation metrics
mae_lesser_known = mean_absolute_error(y_test_all.iloc[lesser_known_indices], y_pred_lesser_known)
mse_lesser_known = mean_squared_error(y_test_all.iloc[lesser_known_indices], y_pred_lesser_known)
r2_lesser_known = r2_score(y_test_all.iloc[lesser_known_indices], y_pred_lesser_known)

# Print evaluation metrics
print("Evaluation metrics for lesser-known brands:")
print("Mean Absolute Error (MAE):", mae_lesser_known)
print("Mean Squared Error (MSE):", mse_lesser_known)
print("R-squared (R2) Score:", r2_lesser_known)
# 3 brand and their significant its price
print("3 brand and their significant its price")
import matplotlib.pyplot as plt
brand_price_averages = df1.groupby('Company')['Price'].mean()
top_30_brands = brand_price_averages.head(19).index
plt.figure(figsize=(10, 6))
brand_price_averages[top_30_brands].plot(kind='bar', color='skyblue')
plt.title("Average Laptop Price by Brand (Top 19)")
plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


#5 limitation this model

print("""
5 quetion This model excels at predicting prices accurately within the dataset, 
including specific laptops, even for future data. However, 
it's important to note that its effectiveness may be limited to these specific laptop types.""")
#6 How does the model perform when predicting the prices of newly released laptops not present in the training dataset?
# Get the feature names used during training
feature_names = x_train_all.columns

# Create a dictionary for the new observation with 338 features
print("this is my new laptop and its predictions")
new_observation_data = {
    'Company_Acer': [1],  # Example: Use 1 if it belongs to Apple, 0 otherwise
    'one 14': [1],  # Example: Use 1 if it is an Ultrabook, 0 otherwise
    'Inches': [13.3],  # Example: Screen size in inches
    'ScreenResolution_FHD Display 1920x1080': [1],  # Example: Use 1 if it has this screen resolution, 0 otherwise
    'Cpu_Intel Core i7 4.4GHz': [1],  # Example: Use 1 if it has this CPU, 0 otherwise
    'Ram': [8],  # Example: RAM size in GB
    'Memory_512GB SSD': [1],  # Example: Use 1 if it has this memory configuration, 0 otherwise
    'Intel Integrated Integrated': [1],  # Example: Use 1 if it has this GPU, 0 otherwise
    'Windows 11': [1],  # Example: Use 1 if it has macOS, 0 otherwise
    'Weight': [1.49]  # Example: Weight of the laptop
}

# Create a DataFrame for the new observation
new_observation_df = pd.DataFrame(new_observation_data, columns=feature_names)

# Scale the features of the new observation using the same scaler used during training
new_observation_scaled = scaler.transform(new_observation_df)

# Make predictions for the new observation using the trained pipeline
y_pred_new_observation = pipeline_all.predict(new_observation_scaled)

# Print the predicted price for the new observation
print("Predicted price for the new observation:", y_pred_new_observation)
top_brands = laptop.groupby('Company')['Price'].sum().nlargest(10)

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_brands.index, top_brands.values)
plt.xlabel("Brand")
plt.ylabel("Total Sales")
plt.title("Top 10 Brands by Total Sales")
plt.xticks(rotation=45)
plt.show() 
