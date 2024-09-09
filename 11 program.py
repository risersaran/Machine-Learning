# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset (Assuming it's a CSV file)
# The dataset should have columns like 'income', 'loan_amount', 'num_defaults', 'credit_score' (target)
# Replace 'your_dataset.csv' with the actual dataset path
df = pd.read_csv('your_dataset.csv')

# Preview the data
print(df.head())

# Preprocess the dataset
# Handling missing values
df.fillna(df.mean(), inplace=True)

# Encoding the target variable (Credit score: 'Good', 'Average', 'Poor')
label_encoder = LabelEncoder()
df['credit_score'] = label_encoder.fit_transform(df['credit_score'])

# Define features and target variable
X = df[['income', 'loan_amount', 'num_defaults']]  # Features
y = df['credit_score']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
