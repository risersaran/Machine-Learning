import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create a larger sample dataset
data = {
    'size': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    'bedrooms': [3, 4, 3, 5, 4, 5, 4, 6, 5, 6],
    'bathrooms': [2, 3, 2, 4, 3, 4, 3, 5, 4, 5],
    'floors': [1, 2, 1, 2, 2, 3, 2, 3, 3, 3],
    'age': [10, 15, 20, 5, 8, 6, 7, 10, 12, 14],
    'price': [400000, 500000, 600000, 700000, 800000, 850000, 900000, 950000, 1000000, 1050000]
}

# Load the dataset into a DataFrame
df = pd.DataFrame(data)

# Preview the dataset
print(df.head())

# Basic information about the dataset
print(df.info())

# Define feature columns and target variable
X = df[['size', 'bedrooms', 'bathrooms', 'floors', 'age']]  # Features (input variables)
y = df['price']  # Target variable (house price)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (normalizing the features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model on the training set
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Compare some predicted prices with actual prices
comparison_df = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred})
print(comparison_df.head())
