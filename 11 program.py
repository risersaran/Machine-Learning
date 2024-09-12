import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'C:/Users/dinak/Downloads/income.csv'
df = pd.read_csv(file_path)

# Preview the data
print("Data preview:")
print(df.head())

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include='number').columns
non_numeric_cols = df.select_dtypes(exclude='number').columns

# Handle missing values for numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Ensure the target column exists
target_column = 'credit_score'  # Update this if the target column is named differently
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")

# Encoding the target variable
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# Define features and target variable
X = df[['income', 'loan_amount', 'num_defaults']]  # Features (ensure these columns exist)
y = df[target_column]  # Target variable

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

# Determine unique classes in y_test and y_pred
unique_classes = sorted(set(y_test) | set(y_pred))
target_names = label_encoder.inverse_transform(unique_classes)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=target_names))

# Print unique values in y_test and y_pred for debugging
print("Unique values in y_test:", set(y_test))
print("Unique values in y_pred:", set(y_pred))
