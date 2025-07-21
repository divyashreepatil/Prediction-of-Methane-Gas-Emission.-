#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Step 1: Import necessary libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Step 2: Define dataset path (Update if needed)
dataset_path = "./extracted_dataset/FAOSTAT_data_en_11-14-2023.csv"  # Adjust path if different

# Step 3: Load Dataset
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print("Dataset Loaded Successfully!")
else:
    print("Error: Dataset not found! Check the file path.")

# Step 4: Preprocess Data
# Selecting features and target variable
features = ["Area Code (M49)", "Element Code", "Item Code", "Year", "Source Code"]
target = "Value"

# Drop rows with missing values in target column
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train KNN Model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Step 8: Make Predictions
y_pred_knn = knn_model.predict(X_test_scaled)

# Step 9: Calculate R² Score
r2_knn = r2_score(y_test, y_pred_knn)
print("R² Score for KNN:", r2_knn)


# In[3]:


for k in range(1, 20):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print(f"K={k}, R² Score: {r2_score(y_test, y_pred)}")


# In[17]:


# Import necessary libraries
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Step 1: Define File Paths
zip_file_path = "./FAOSTAT_data_en_11-14-2023.csv.zip"  # Ensure correct file path
extract_folder = "./extracted_dataset"

# Step 2: Unzip File if Not Already Extracted
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

# Step 3: Load Dataset
csv_file_path = os.path.join(extract_folder, "FAOSTAT_data_en_11-14-2023.csv")  # Update if different
df = pd.read_csv(csv_file_path)

# Step 4: Debug - Print first few rows
print("\n First 5 rows of the dataset:")
print(df.head())

# Step 5: Check column names
print("\n Available Columns:", df.columns.tolist())

# Step 6: Ensure "Value" column exists
if "Value" not in df.columns:
    raise ValueError(" ERROR: 'Value' column is missing from the dataset!")

# Step 7: Remove unnecessary categorical columns
columns_to_exclude = ["Area", "Element", "Item", "Source", "Note"]  # Adjust as needed
df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns], errors="ignore")

# Step 8: Select only numerical columns
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Ensure "Value" exists and remove it from features
if "Value" in numeric_features:
    numeric_features.remove("Value")


# Step 10: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 12: Train Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42)
gbr_model.fit(X_train_scaled, y_train)

# Step 13: Predictions
y_pred = gbr_model.predict(X_test_scaled)

# Step 14: Model Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Step 15: Print Results
print(f"\n R² Score (GBR): {r2:.4f}")
print(f" Mean Absolute Error (MAE): {mae:.2f}")
print(f" Root Mean Square Error (RMSE): {rmse:.2f}")


# In[23]:


# Step 14: Model Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Step 15: Print Results
print(f"\n R² Score (GBR): {r2:.4f}")
print(f" Root Mean Square Error (RMSE): {rmse:.2f}")


# In[ ]:




