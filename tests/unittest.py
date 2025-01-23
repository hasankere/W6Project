import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Sample DataFrame for testing
class TestDataProcessingAndModeling(unittest.TestCase):

    def setUp(self):
        """Setup a sample DataFrame with missing values and features."""
        # Sample DataFrame with some missing values
        self.data = pd.DataFrame({
            'Amount': [100, np.nan, 300, 400, np.nan],
            'Value': [50, 70, np.nan, 90, 110],
            'ProductCategory': ['A', 'B', np.nan, 'A', 'C'],
            'CountryCode': ['US', np.nan, 'UK', 'US', 'IN'],
            'RiskCategory': [0, 1, 0, 1, 0]
        })
        self.numerical_features = ['Amount', 'Value']
        self.categorical_features = ['ProductCategory', 'CountryCode']
        self.target = 'RiskCategory'

        # Split data into features (X) and target (y)
        self.X = self.data[self.numerical_features + self.categorical_features]
        self.y = self.data[self.target]
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )

        # Create a model pipeline
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    def test_fill_missing_values_numerical(self):
        """Test that missing numerical values are filled with the median."""
        # Fill missing numerical values with the median
        imputer = SimpleImputer(strategy='median')
        self.X[self.numerical_features] = imputer.fit_transform(self.X[self.numerical_features])
        
        # Assert that no missing values exist in numerical columns
        for feature in self.numerical_features:
            self.assertEqual(self.X[feature].isnull().sum(), 0)

    def test_fill_missing_values_categorical(self):
        """Test that missing categorical values are filled with the mode."""
        # Fill missing categorical values with the mode
        imputer = SimpleImputer(strategy='most_frequent')
        self.X[self.categorical_features] = imputer.fit_transform(self.X[self.categorical_features])
        
        # Assert that no missing values exist in categorical columns
        for feature in self.categorical_features:
            self.assertEqual(self.X[feature].isnull().sum(), 0)

    def test_preprocessing_pipeline(self):
        """Test that the preprocessing pipeline works without errors."""
        # Fit and transform the features through the pipeline
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)
        
        # Check if the transformations are successful (no error)
        self.assertEqual(self.X_train_transformed.shape[0], self.X_train.shape[0])
        self.assertEqual(self.X_test_transformed.shape[0], self.X_test.shape[0])

    def test_model_training(self):
        """Test that the model can be trained without errors."""
        # Train the model pipeline
        self.model_pipeline.fit(self.X_train, self.y_train)
        
        # Check if the model has been trained (i.e., the classifier is not None)
        self.assertIsNotNone(self.model_pipeline.named_steps['classifier'])

    def test_model_evaluation(self):
        """Test that the model evaluation metrics are within a reasonable range."""
        # Make predictions and calculate probabilities
        y_pred = self.model_pipeline.predict(self.X_test)
        y_pred_proba = self.model_pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate evaluation metrics
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        # Assert that the metrics are within a reasonable range (e.g., 0.5-1 for AUC, accuracy, etc.)
        self.assertGreater(auc_roc, 0.5)
        self.assertGreater(accuracy, 0.5)
        self.assertGreater(precision, 0.5)
        self.assertGreater(recall, 0.5)
        self.assertGreater(f1, 0.5)

    def test_confusion_matrix(self):
        """Test that the confusion matrix can be generated without errors."""
        # Get confusion matrix
        y_pred = self.model_pipeline.predict(self.X_test)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Assert that confusion matrix has correct dimensions (2x2 for binary classification)
        self.assertEqual(conf_matrix.shape, (2, 2))

    def test_classification_report(self):
        """Test that the classification report is generated correctly."""
        # Get classification report
        y_pred = self.model_pipeline.predict(self.X_test)
        class_report = classification_report(self.y_test, y_pred)
        
        # Ensure classification report contains the correct labels
        self.assertIn('precision', class_report)
        self.assertIn('recall', class_report)
        self.assertIn('f1-score', class_report)

if __name__ == '__main__':
    unittest.main()
