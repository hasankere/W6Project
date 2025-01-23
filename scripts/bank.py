# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
# File path to the CSV
file_path = 'C:\\Users\\Hasan\\Desktop\\data science folder\\data.csv'

# Load the CSV without headers
data = pd.read_csv(file_path)
print(data.head())

# Step 1: Define Proxy Variable for Risk (FraudRate)
# Aggregating fraud rate per customer
customer_risk = data.groupby('CustomerId')['FraudResult'].mean().reset_index()
customer_risk.columns = ['CustomerId', 'CustomerFraudRate']

# Define High-Risk vs Low-Risk
threshold = 0.2  # Example threshold for fraud rate
customer_risk['RiskCategory'] = (customer_risk['CustomerFraudRate'] > threshold).astype(int)

# Merge with original data
data = data.merge(customer_risk[['CustomerId', 'RiskCategory']], on='CustomerId', how='left')

# Step 2: Select Features for Predicting Risk
# Define feature columns
categorical_features = ['ChannelId', 'ProductCategory', 'CurrencyCode', 'CountryCode']
numerical_features = ['Amount', 'Value', 'TransactionStartTime']
target = 'RiskCategory'
# Step 3: Train Risk Probability Model
# Splitting the data
X = data[categorical_features + numerical_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: OneHotEncoding for categorical and StandardScaler for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report

# Assuming 'data' is your DataFrame, and 'TransactionStartTime' is a datetime column

# Ensure 'TransactionStartTime' is in datetime format
data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

# Create new features
data['TransactionHour'] = data['TransactionStartTime'].dt.hour
data['TransactionDayOfWeek'] = data['TransactionStartTime'].dt.dayofweek
data['IsWeekend'] = data['TransactionDayOfWeek'].isin([5, 6]).astype(int)  # 5 = Saturday, 6 = Sunday

# Define numerical features (after adding new columns)
numerical_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDayOfWeek', 'IsWeekend']

# Define categorical features
categorical_features = ['ChannelId', 'ProductCategory', 'CurrencyCode', 'CountryCode']

# Define the target
target = 'RiskCategory'

# Splitting the data into features and target variable
X = data[categorical_features + numerical_features]
y = data[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Model pipeline: first preprocessing, then classifier
risk_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
risk_model.fit(X_train, y_train)

# Predict probabilities and class labels on the test set
y_pred_proba = risk_model.predict_proba(X_test)[:, 1]
y_pred = risk_model.predict(X_test)

# Evaluate the model using different metrics
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Overview of the dataset
print("Dataset Overview:")
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")
print("\nData Types:")
print(data.dtypes)

# Display the first few rows
print("\nFirst 5 Rows:")
print(data.head())
# Summary statistics for numerical columns
print("\nSummary Statistics (Numerical Features):")
print(data.describe())

# Summary statistics for categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print("\nSummary Statistics (Categorical Features):")
    print(data[categorical_columns].describe())
else:
    print("\nNo categorical features detected in the dataset.")
#Distribution of Numerical Features
import seaborn as sns
import matplotlib.pyplot as plt

# Numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Plot distributions
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
# Display the distribution of categorical features
for feature in categorical_features:
    print(f"\nDistribution of {feature}:")
    print(data[feature].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns
#Distribution of categorical variable
# Set plot style
sns.set(style="whitegrid")

# Plot the distribution of each categorical feature
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=feature, palette="Set2")
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.show()
# Handling High Cardinality Features
# Plot the distribution of each categorical feature, but limit to the top N most frequent categories
top_n = 10  # You can adjust this number based on your preference

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    top_categories = data[feature].value_counts().nlargest(top_n).index
    sns.countplot(data=data, x=feature, order=top_categories, palette="Set2")
    plt.title(f'Distribution of {feature} (Top {top_n})', fontsize=16)
    plt.xticks(rotation=45)
    plt.show()
# Display percentage distribution for each categorical feature
for feature in categorical_features:
    print(f"\nPercentage distribution of {feature}:")
    print(data[feature].value_counts(normalize=True) * 100)
# Compute correlation matrix for numerical features
correlation_matrix = data[numerical_features].corr()

# Display the correlation matrix
print(correlation_matrix)
# Set a threshold for high correlation (e.g., 0.7 or -0.7)
threshold = 0.7

# Find pairs of highly correlated features
highly_correlated = correlation_matrix.abs() > threshold
for feature in highly_correlated.columns:
    correlated_features = highly_correlated[feature].index[highly_correlated[feature]].tolist()
    if len(correlated_features) > 1:
        print(f"\nHighly correlated with {feature}: {', '.join(correlated_features)}")
# Pairwise scatter plots for numerical features
sns.pairplot(data[numerical_features])
plt.suptitle('Pairwise Scatter Plots of Numerical Features', fontsize=16)
plt.show()
# Check for missing values
print("\nMissing Values:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Visualize missing values
import missingno as msno
plt.figure(figsize=(10, 6))
msno.matrix(data)
plt.title("Missing Values Heatmap")
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Create box plots for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=feature, palette="Set2")
    plt.title(f'Box Plot for {feature}', fontsize=16)
    plt.show()
# Calculate the IQR for each numerical feature
for feature in numerical_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find the outliers
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    
    print(f"\nOutliers for {feature}:")
    print(outliers[feature].head())  # Display the first few outliers

    # Optionally, display the count of outliers
    print(f"Number of outliers for {feature}: {len(outliers)}")
# Check for missing values in the dataset
missing_values = data.isnull().sum()
print(f"Missing values in each feature:\n{missing_values}")
# Fill missing values for numerical features with the median
for feature in numerical_features:
    data[feature].fillna(data[feature].median(), inplace=True)

# Fill missing values for categorical features with the mode (most frequent category)
for feature in categorical_features:
    data[feature].fillna(data[feature].mode()[0], inplace=True)
# Visualize missing values before imputation
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
plt.title("Missing Values Before Imputation", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel('Number of Missing Values')
plt.show()
# Check for missing values after imputation
missing_values_after = data.isnull().sum()

# Visualize missing values after imputation
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values_after.index, y=missing_values_after.values, palette='viridis')
plt.title("Missing Values After Imputation", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel('Number of Missing Values')
plt.show()
