# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate a synthetic dataset
np.random.seed(42)

# Create a sample dataset with 1000 records
data = {
    'age': np.random.randint(1, 15, 1000),  # Age of the car
    'mileage': np.random.randint(5000, 200000, 1000),  # Mileage in kilometers
    'engine_size': np.random.uniform(1.0, 4.5, 1000),  # Engine size in liters
    'horsepower': np.random.randint(80, 400, 1000),  # Horsepower
    'make': np.random.choice(['Toyota', 'Honda', 'BMW', 'Ford', 'Audi'], 1000),  # Car manufacturer
    'model': np.random.choice(['Model_A', 'Model_B', 'Model_C', 'Model_D'], 1000),  # Car model
    'price': np.random.randint(5000, 50000, 1000)  # Price in dollars
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Preview the dataset
print(df.head())

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Encode categorical features (e.g., 'make', 'model')
label_encoder = LabelEncoder()
df['make'] = label_encoder.fit_transform(df['make'])
df['model'] = label_encoder.fit_transform(df['model'])

# Define the feature columns and target variable
X = df[['age', 'mileage', 'engine_size', 'horsepower', 'make', 'model']]  # Features
y = df['price']  # Target variable (car price)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Compare some predicted prices with actual prices
comparison_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print(comparison_df.head())
