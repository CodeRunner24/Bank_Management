# Bank Marketing Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Read the data from bank-additional
data = pd.read_csv('bank-additional/bank-additional.csv', sep=';')


# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Check data types and missing values
print("\nData types:")
print(data.dtypes)
print("\nMissing values count:")
print(data.isnull().sum())

# Check for "unknown" values in columns
print("\nCounting 'unknown' values in categorical columns:")
for col in data.select_dtypes(include=['object']).columns:
    unknown_count = (data[col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"{col}: {unknown_count} unknown values ({unknown_count / len(data) * 100:.2f}%)")

# Check the target variable distribution
print("\nTarget variable distribution:")
print(data['y'].value_counts())
print(data['y'].value_counts(normalize=True) * 100)

# Visualize the target variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=data)
plt.title('Target Variable Distribution')
plt.ylabel('Count')
plt.savefig('target_distribution.png')
plt.close()

# Explore age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='y', kde=True, multiple='stack')
plt.title('Age Distribution by Target')
plt.savefig('age_distribution.png')
plt.close()

# Explore categorical variables
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']

for col in categorical_cols[:5]:  # Only showing first 5 to save space
    plt.figure(figsize=(12, 6))
    sns.countplot(x=col, hue='y', data=data)
    plt.title(f'{col} Distribution by Target')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{col}_distribution.png')
    plt.close()

# Explore numerical variables
numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                  'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Create correlation matrix for numerical variables
plt.figure(figsize=(14, 10))
correlation_matrix = data[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

#--------------------------------
# DATA CLEANING AND PREPROCESSING
#--------------------------------

# Replace 'unknown' values with NaN
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].replace('unknown', np.nan)

# Fix issues with numeric columns that might be mistakenly read as strings
# Check for potential numeric columns with string values
for col in ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']:
    if data[col].dtype == 'object':
        # Replace comma with dot for decimal separator
        data[col] = data[col].str.replace(',', '.').astype(float)

# Check for special values in pdays (999 means client was not previously contacted)
print("\nUnique values in pdays:", data['pdays'].unique())

# Handle 'pdays' feature - create a binary feature for "was contacted before"
data['was_contacted_before'] = (data['pdays'] != 999).astype(int)

# Fix 'y' target variable (convert to binary)
data['y'] = (data['y'] == 'yes').astype(int)

# Define feature columns
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']
numerical_features = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate',
                      'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'was_contacted_before']

# Split data into features and target
X = data.drop('y', axis=1)
y = data['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#-----------------------------------------------
# FEATURE ENGINEERING AND PREPROCESSING PIPELINE
#-----------------------------------------------

# Create preprocessor for categorical and numerical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

#-------------------------------
# MODEL SELECTION AND EVALUATION
#-------------------------------

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Dictionary to store results
results = {}

# Evaluate models with cross-validation
for name, model in models.items():
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')

    # Store results
    results[name] = {
        'cv_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std()
    }

    print(f"{name} - Mean ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Identify the best model based on cross-validation
best_model_name = max(results, key=lambda x: results[x]['mean_cv_score'])
print(f"\nBest model: {best_model_name} with ROC-AUC: {results[best_model_name]['mean_cv_score']:.4f}")

#-----------------------------------------
# HYPERPARAMETER TUNING FOR THE BEST MODEL
#-----------------------------------------

# Define hyperparameter grid for the best model
if best_model_name == 'Logistic Regression':
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
else:  # Gradient Boosting
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }

# Create a pipeline with the best model
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', models[best_model_name])
])

# Perform grid search
grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters
print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

#-----------------------
# FINAL MODEL EVALUATION
#-----------------------

# Get the best model
final_model = grid_search.best_estimator_

# Make predictions on test set
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = (y_pred == y_test).mean()
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nFinal Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')
plt.close()

#----------------------------
# FEATURE IMPORTANCE ANALYSIS
#----------------------------

# Extract feature names after preprocessing
preprocessor.fit(X_train)
feature_names = []

# Extract column names from OneHotEncoder
ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
    categorical_features)
for name in ohe_feature_names:
    feature_names.append(name)

# Add numerical feature names
for name in numerical_features:
    feature_names.append(name)

# Get feature importances (for tree-based models)
if hasattr(final_model.named_steps['model'], 'feature_importances_'):
    importances = final_model.named_steps['model'].feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

    print("\nTop 10 important features:")
    for i in range(min(10, len(indices))):
        print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

#---------------
# SAVE THE MODEL
#---------------

# Save the model for streamlit app
print("\nSaving model as 'bank_marketing_model.pkl'...")
joblib.dump(final_model, 'bank_marketing_model.pkl')
print("Model saved successfully!")


#-------------------------
# Function for prediction (to be used in Streamlit app)
#-------------------------

def predict_subscription(input_data):
    """
    Make prediction on new client data.
    input_data: DataFrame with client information
    returns: probability of subscription and binary prediction
    """
    # Load the model
    model = joblib.load('bank_marketing_model.pkl')
    
    # Make prediction
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]

    return prediction_proba, prediction

if __name__ == "__main__":
    print("\nModel training and evaluation completed successfully.")
    print("You can now run the Streamlit app with: streamlit run streamlit.py")