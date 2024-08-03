import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
import joblib
import logging
from tqdm import tqdm
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
import torch
from time import time

print(torch.__version__)
print(torch.version.cuda)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load the preprocessed data."""
    return pd.read_csv('feature_data/listings_with_engineered_features.csv')

def prepare_features(df):
    """Prepare features and handle categorical variables."""
    # Remove unnecessary columns
    columns_to_drop = ['name', 'host_name', 'last_review', 'license', 'price_category']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove 'price' from features if present
    if 'price' in numeric_features:
        numeric_features.remove('price')
    
    features = numeric_features + categorical_features
    
    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorical features: {categorical_features}")
    
    X = df[features]
    y = np.log1p(df['price'])  # Log-transform the target variable
    return X, y, numeric_features, categorical_features

def create_model_pipeline(numeric_features, categorical_features, use_gpu):
    """Create a pipeline that includes preprocessing and the model."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define base models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, 
                                 tree_method='gpu_hist' if use_gpu else 'hist')
    lgbm = LGBMRegressor(random_state=42, device='gpu' if use_gpu else 'cpu')

    # Create stacking model
    stacking_model = StackingRegressor(
        estimators=[('rf', rf), ('xgb', xgb_model), ('lgbm', lgbm)],
        final_estimator=Lasso(random_state=42),
        cv=5
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', RFE(estimator=RandomForestRegressor(n_estimators=10, random_state=42), n_features_to_select=20)),
        ('model', stacking_model)
    ])

    return pipeline

def train_and_evaluate_model(X, y, numeric_features, categorical_features, use_gpu):
    """Train and evaluate the model using cross-validation and hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = create_model_pipeline(numeric_features, categorical_features, use_gpu)
    
    # Define hyperparameter search space
    param_distributions = {
        'model__rf__n_estimators': [100, 200, 300],
        'model__xgb__n_estimators': [100, 200, 300],
        'model__xgb__learning_rate': [0.01, 0.1, 0.3],
        'model__lgbm__n_estimators': [100, 200, 300],
        'model__lgbm__learning_rate': [0.01, 0.1, 0.3],
        'model__final_estimator__alpha': [0.1, 1, 10]
    }
    
    # Perform randomized search
    n_iter = 10
    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=n_iter, cv=5, n_jobs=-1, random_state=42, verbose=2)
    
    # Fit the model with progress bar
    start_time = time()
    try:
        logging.info("Starting hyperparameter tuning...")
        random_search.fit(X_train, y_train)
        logging.info("Hyperparameter tuning completed.")
    except Exception as e:
        logging.error(f"An error occurred during hyperparameter tuning: {str(e)}")
        raise

    # Get best model
    best_model = random_search.best_estimator_
    
    # Evaluate on the test set
    logging.info("Evaluating best model on test set...")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Model Performance:")
    logging.info(f"Test MSE: {mse:.4f}")
    logging.info(f"Test R2: {r2:.4f}")
    
    return best_model

# Update the main function to include more logging
def main():
    # Check for GPU availability
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info("GPU is available. Using GPU for training.")
    else:
        logging.info("GPU is not available. Using CPU for training.")

    logging.info("Loading data...")
    df = load_data()
    logging.info("Preparing features...")
    X, y, numeric_features, categorical_features = prepare_features(df)
    
    logging.info("Starting model training and evaluation...")
    best_model = train_and_evaluate_model(X, y, numeric_features, categorical_features, use_gpu)
    
    logging.info("Saving best model...")
    save_model(best_model, 'models/best_price_prediction_model.joblib')
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()