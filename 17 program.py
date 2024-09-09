# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'mobile_price_data.csv' with the actual file path)
# Example features: 'battery_power', 'clock_speed', 'ram', 'internal_storage', 'screen_size', 'mobile_price_range'
df = pd.read_csv('mobile_price_data.csv')

# Preview the dataset
print(df.head())

# Basic information about the dataset
print(df.info())

# Handling missing values (if any)
df.fillna(df.mean(), inplace=True)

# Define features and target
X = df.drop(columns=['price_range'])  # Features (excluding the target)
y = df['price_range']  # Target variable (price range)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the classification report for detailed performance analysis
print('Classification Report:')
print(classification_report(y_test, y_pred))
