# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For data preprocessing and systems 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

# Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# For saving the model
import joblib

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Check and print package versions
import sklearn
import xgboost
import lightgbm

print(f"scikit-learn version: {sklearn.__version__}")
print(f"xgboost version: {xgboost.__version__}")
print(f"lightgbm version: {lightgbm.__version__}")

# Load the dataset
df = pd.read_csv('rdata.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Get information about data types and missing values
print("\nDataset Information:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Separate features by data type
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Impute numerical features with median
numerical_imputer = SimpleImputer(strategy='median')

# Impute categorical features with mode
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputers
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# Remove duplicates
df.drop_duplicates(inplace=True)
print(f"Dataset shape after removing duplicates: {df.shape}")

# Handle erroneous data (e.g., negative prices)
# Ensure 'price' is the target variable
if 'price' in df.columns:
    initial_shape = df.shape
    df = df[df['price'] >= 0]
    final_shape = df.shape
    print(f"\nDataset shape after removing negative prices: {final_shape[0] - initial_shape[0]} rows removed")
else:
    print("\nError: 'price' column not found in the dataset.")
    exit()

# Outlier Detection and Removal using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in numerical_cols:
    if col == 'price':
        continue  # Typically, don't remove outliers based on the target variable
    initial_shape = df.shape
    df = remove_outliers_iqr(df, col)
    final_shape = df.shape
    removed = initial_shape[0] - final_shape[0]
    if removed > 0:
        print(f"Removed outliers from {col}: {removed} rows removed")

# Feature Engineering (Modify based on your dataset)
# Example:
# if 'date' in df.columns:
#     df['year'] = pd.to_datetime(df['date']).dt.year
#     df['month'] = pd.to_datetime(df['date']).dt.month
#     df.drop('date', axis=1, inplace=True)

# Encoding Categorical Variables using One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"\nDataset shape after encoding categorical variables: {df.shape}")

# Feature Scaling using StandardScaler
scaler = StandardScaler()
# Exclude the target variable from scaling
features_to_scale = numerical_cols.copy()
features_to_scale.remove('price')

df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Define target and features
X = df.drop('price', axis=1)
y = df['price']

# Handle Multicollinearity
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

print(f"\nFeatures to drop due to multicollinearity: {to_drop}")

# Drop features
X = X.drop(columns=to_drop)
print(f"Dataset shape after dropping correlated features: {X.shape}")

# Polynomial Features (Degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X)

# Update Feature Names
poly_feature_names = poly.get_feature_names_out(X.columns)
X_poly = pd.DataFrame(X_poly, columns=poly_feature_names)

print(f"\nDataset shape after adding polynomial features: {X_poly.shape}")

# Define target and features
X = X_poly
y = y  # Target remains the same

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define Cross-Validation Strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to evaluate models with cross-validation
def evaluate_model_cv(model, X, y, cv):
    """
    Perform cross-validation, train the model, make predictions, and evaluate performance.
    Returns a dictionary of evaluation metrics and CV score.
    """
    # Cross-Validation Score
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    cv_score = cv_scores.mean()
    
    # Train-Test Split Evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'CV_R2': cv_score
    }

# Dictionary to store model performance
model_performance = {}

# Define Selected Models
selected_models = {
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
}

# Evaluate Selected Models
for name, model in selected_models.items():
    print(f"Training and evaluating {name}...")
    results = evaluate_model_cv(model, X, y, cv=kf)
    model_performance[name] = results

# Convert to DataFrame
performance_df = pd.DataFrame(model_performance).T
performance_df = performance_df[['MAE', 'MSE', 'RMSE', 'R2', 'CV_R2']]

# Sort by R2 Score in Descending Order
performance_df_sorted = performance_df.sort_values(by='R2', ascending=False)

# Display Performance
print("\nModel Performance Comparison:")
print(performance_df_sorted)

# Visualize R2 Scores
plt.figure(figsize=(10, 6))
sns.barplot(x='R2', y=performance_df_sorted.index, data=performance_df_sorted.reset_index())
plt.title('R2 Score Comparison of Selected Regression Models')
plt.xlabel('R2 Score')
plt.ylabel('Regression Models')
plt.xlim(0, 1)
plt.show()

# Visualize RMSE
plt.figure(figsize=(10, 6))
sns.barplot(x='RMSE', y=performance_df_sorted.index, data=performance_df_sorted.reset_index())
plt.title('RMSE Comparison of Selected Regression Models')
plt.xlabel('RMSE')
plt.ylabel('Regression Models')
plt.show()

# Summary of Results
print("\nSummary of Model Performance (Sorted by R2 Score):")
for index, row in performance_df_sorted.iterrows():
    print(f"\nModel: {index}")
    print(f"MAE: {row['MAE']:.2f}")
    print(f"MSE: {row['MSE']:.2f}")
    print(f"RMSE: {row['RMSE']:.2f}")
    print(f"R2 Score: {row['R2']:.4f}")
    print(f"Cross-Validated R2 Score: {row['CV_R2']:.4f}")

# Hyperparameter Tuning for the Best Model (Assuming Random Forest is Best)
best_model_name = performance_df_sorted.index[0]
best_model = selected_models[best_model_name]

if best_model_name == 'Random Forest Regression':
    print(f"\nStarting Hyperparameter Tuning for {best_model_name}...")
    rf_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=rf_param_grid,
        scoring='r2',
        cv=kf,
        verbose=1,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    print(f"\nBest {best_model_name} Parameters: {rf_grid.best_params_}")
    tuned_rf = rf_grid.best_estimator_
    tuned_rf_results = evaluate_model_cv(tuned_rf, X, y, cv=kf)
    print(f"\nTuned {best_model_name} Performance:")
    print(tuned_rf_results)
    
    # Save the Tuned Model
    joblib.dump(tuned_rf, 'tuned_random_forest_model.pkl')
    print("\nTuned Random Forest model saved as 'tuned_random_forest_model.pkl'")

elif best_model_name == 'Gradient Boosting Regression':
    print(f"\nStarting Hyperparameter Tuning for {best_model_name}...")
    gb_param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5],
        'subsample': [0.7, 0.9],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    gb_grid = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=gb_param_grid,
        scoring='r2',
        cv=kf,
        verbose=1,
        n_jobs=-1
    )
    gb_grid.fit(X_train, y_train)
    print(f"\nBest {best_model_name} Parameters: {gb_grid.best_params_}")
    tuned_gb = gb_grid.best_estimator_
    tuned_gb_results = evaluate_model_cv(tuned_gb, X, y, cv=kf)
    print(f"\nTuned {best_model_name} Performance:")
    print(tuned_gb_results)
    
    # Save the Tuned Model
    joblib.dump(tuned_gb, 'tuned_gradient_boosting_model.pkl')
    print("\nTuned Gradient Boosting model saved as 'tuned_gradient_boosting_model.pkl'")

elif best_model_name == 'Support Vector Regression':
    print(f"\nStarting Hyperparameter Tuning for {best_model_name}...")
    svr_param_grid = {
        'C': [50, 100, 150],
        'gamma': [0.05, 0.1, 0.2],
        'epsilon': [0.05, 0.1, 0.2]
    }
    svr_grid = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid=svr_param_grid,
        scoring='r2',
        cv=kf,
        verbose=1,
        n_jobs=-1
    )
    svr_grid.fit(X_train, y_train)
    print(f"\nBest {best_model_name} Parameters: {svr_grid.best_params_}")
    tuned_svr = svr_grid.best_estimator_
    tuned_svr_results = evaluate_model_cv(tuned_svr, X, y, cv=kf)
    print(f"\nTuned {best_model_name} Performance:")
    print(tuned_svr_results)
    
    # Save the Tuned Model
    joblib.dump(tuned_svr, 'tuned_svr_model.pkl')
    print("\nTuned Support Vector Regression model saved as 'tuned_svr_model.pkl'")
