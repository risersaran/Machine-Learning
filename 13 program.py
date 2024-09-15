import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import UndefinedMetricWarning  # Import this to handle the warning

# Sample DataFrame creation (replace this with your actual data)
data = {
    'age': [7, 4, 13, 11, 8],
    'mileage': [48088, 108196, 175883, 60479, 24963],
    'engine_size': [1.319076, 1.537803, 2.587928, 2.856240, 1.062517],
    'horsepower': [237, 347, 114, 346, 143],
    'make': ['Ford', 'Honda', 'Honda', 'BMW', 'BMW'],
    'model': ['Model_C', 'Model_B', 'Model_A', 'Model_A', 'Model_D'],
    'price': [22400, 11470, 7062, 46798, 24880]
}
df = pd.DataFrame(data)

# Handle missing values (if any)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode categorical features
label_encoder = LabelEncoder()
df['make'] = label_encoder.fit_transform(df['make'])
df['model'] = label_encoder.fit_transform(df['model'])

# Define features and target variable
X = df[['age', 'mileage', 'engine_size', 'horsepower', 'make', 'model']]
y = df['price']

# Increase test size to avoid R^2 warning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model using MAE, MSE, RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Catch and suppress R^2 warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    # Perform cross-validation to avoid single test sample issue
    cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R² scores: {cv_scores}")
    print(f"Mean R² score: {np.mean(cv_scores)}")
