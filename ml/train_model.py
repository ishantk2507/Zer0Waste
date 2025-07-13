import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

MODEL_PATH = "ml/spoilage_model.joblib"
DATA_PATH = "data/mock_inventory.csv"

def safe_divide(a, b, default=0.0):
    """Safe division handling divide by zero"""
    try:
        if isinstance(b, (pd.Series, np.ndarray)):
            mask = (b != 0)
            result = np.zeros_like(a, dtype=float)
            result[mask] = a[mask] / b[mask]
            result[~mask] = default
            return result
        return a / b if b != 0 else default
    except Exception:
        return default

def engineer_features(df):
    """Enhanced feature engineering with domain-specific knowledge and edge case handling"""
    
    # Time-based features with validation
    current_date = pd.to_datetime('today')
    df['stock_date'] = pd.to_datetime(df['stock_date'])
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])
    
    # Ensure dates are valid
    df['stock_date'] = df['stock_date'].fillna(current_date)
    df['expiry_date'] = df['expiry_date'].fillna(current_date + pd.Timedelta(days=7))
    
    # Calculate time-based features with validation
    df['days_old'] = (current_date - df['stock_date']).dt.days.clip(lower=0)
    df['days_to_expiry'] = (df['expiry_date'] - current_date).dt.days.clip(lower=0)
    df['shelf_life'] = (df['expiry_date'] - df['stock_date']).dt.days.clip(lower=1)  # Avoid division by zero
    
    # Normalized time features with safe division
    df['life_used_ratio'] = safe_divide(df['days_old'], df['shelf_life'])
    df['remaining_life_ratio'] = safe_divide(df['days_to_expiry'], df['shelf_life'])
    
    # Temperature features with validation
    storage_temp_ranges = {
        'Frozen': (-20, -15),
        'Refrigerated': (1, 4),
        'Ambient': (15, 25)
    }
    
    # Ensure temperature and humidity are within physical limits
    df['temperature'] = df['temperature'].clip(-30, 50)
    df['humidity'] = df['humidity'].clip(0, 100)
    
    # Calculate temperature deviation with validation
    df['temp_deviation'] = df.apply(lambda row: 
        abs(row['temperature'] - np.mean(storage_temp_ranges.get(row['storage_type'], (15, 25))))
        if row['storage_type'] in storage_temp_ranges else 0, axis=1
    )
    
    # Humidity features with validation
    ideal_humidity_ranges = {
        'Frozen': (30, 50),
        'Refrigerated': (85, 95),
        'Ambient': (60, 75)
    }
    
    df['humidity_deviation'] = df.apply(lambda row: 
        abs(row['humidity'] - np.mean(ideal_humidity_ranges.get(row['storage_type'], (60, 75))))
        if row['storage_type'] in ideal_humidity_ranges else 0, axis=1
    )
    
    # Storage condition features
    df['storage_type'] = df['storage_type'].fillna('Ambient')
    df['is_frozen'] = (df['storage_type'] == 'Frozen').astype(int)
    df['is_refrigerated'] = (df['storage_type'] == 'Refrigerated').astype(int)
    df['is_ambient'] = (df['storage_type'] == 'Ambient').astype(int)
    
    # Environmental risk scores with safe division
    max_temp_dev = df.groupby('storage_type')['temp_deviation'].transform('max')
    max_hum_dev = df.groupby('storage_type')['humidity_deviation'].transform('max')
    
    df['temp_risk_score'] = safe_divide(df['temp_deviation'], max_temp_dev)
    df['humidity_risk_score'] = safe_divide(df['humidity_deviation'], max_hum_dev)
    
    # Category-specific features
    df['category'] = df['category'].fillna('Unknown')
    high_risk_categories = ['Dairy', 'Meat']
    df['is_high_risk_category'] = df['category'].isin(high_risk_categories).astype(int)
    
    # Location-based features
    df['store_location'] = df['store_location'].fillna('Unknown')
    tropical_cities = ['Chennai', 'Mumbai']
    df['is_tropical_location'] = df['store_location'].isin(tropical_cities).astype(int)
    
    # Interaction features with validation
    df['temp_humid_interaction'] = df['temp_deviation'] * df['humidity_deviation']
    df['time_temp_interaction'] = df['life_used_ratio'] * df['temp_risk_score']
    
    # Final validation: replace any remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def create_feature_matrix(df):
    """Create comprehensive feature matrix with robust scaling"""
    
    # Ensure categorical columns exist
    required_cats = ['product', 'category', 'storage_type', 'store_location']
    for col in required_cats:
        if col not in df.columns:
            df[col] = 'Unknown'
    
    # Categorical encodings with validation
    cat_features = {}
    for col in required_cats:
        dummies = pd.get_dummies(df[col], prefix=col.split('_')[0])
        if dummies.empty:
            dummies = pd.DataFrame({f"{col.split('_')[0]}_Unknown": [1] * len(df)})
        cat_features[col] = dummies
    
    # Numerical features that need scaling
    num_features = [
        'days_old', 'days_to_expiry', 'shelf_life', 
        'life_used_ratio', 'remaining_life_ratio',
        'temp_deviation', 'humidity_deviation',
        'temp_risk_score', 'humidity_risk_score',
        'temp_humid_interaction', 'time_temp_interaction'
    ]
    
    # Ensure all numerical features exist
    for feat in num_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Binary features (no scaling needed)
    binary_features = [
        'is_frozen', 'is_refrigerated', 'is_ambient',
        'is_high_risk_category', 'is_tropical_location'
    ]
    
    # Ensure all binary features exist
    for feat in binary_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Scale numerical features with robust handling
    scaler = StandardScaler()
    num_data = df[num_features].fillna(0)
    
    # Handle potential constant features
    scale_mask = num_data.std() != 0
    if scale_mask.any():
        scaled_features = pd.DataFrame(
            scaler.fit_transform(num_data.loc[:, scale_mask]),
            columns=num_data.columns[scale_mask],
            index=df.index
        )
        # Add back constant features
        for col in num_data.columns[~scale_mask]:
            scaled_features[col] = num_data[col]
    else:
        scaled_features = num_data
    
    # Combine all features
    feature_matrix = pd.concat(
        [*cat_features.values(), scaled_features, df[binary_features]],
        axis=1
    )
    
    # Final validation
    feature_matrix = feature_matrix.fillna(0)
    
    return feature_matrix, scaler

def evaluate_model(model, X, y):
    """Comprehensive model evaluation"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }
    
    scores = {}
    for metric_name, scorer in metrics.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        scores[metric_name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std() * 2
        }
    
    return scores

def train_and_save(tune_hyperparams=True):
    """Enhanced training pipeline with improved feature engineering and model tuning"""
    
    # 1. Load and preprocess data
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    
    # 2. Create feature matrix
    X, scaler = create_feature_matrix(df)
    y = df['spoilage']
    
    # 3. Handle class imbalance
    df_majority = df[y == 0]
    df_minority = df[y == 1]
    
    if len(df_minority) < len(df_majority):
        # Upsample minority class with noise addition
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        
        # Add controlled noise to numerical features
        noise_columns = ['temperature', 'humidity']
        for col in noise_columns:
            noise = np.random.normal(0, df_minority[col].std() * 0.05, len(df_minority_upsampled))
            df_minority_upsampled[col] = df_minority_upsampled[col] + noise
            df_minority_upsampled[col] = df_minority_upsampled[col].clip(
                df[col].min(), df[col].max()
            )
        
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
    else:
        df_balanced = df
    
    # Rebuild features with balanced dataset
    X_balanced, _ = create_feature_matrix(df_balanced)
    y_balanced = df_balanced['spoilage']
    
    # 4. Configure and train model
    if tune_hyperparams:
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        model = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_balanced, y_balanced)
        best_model = model.best_estimator_
    else:
        best_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42
        )
        best_model.fit(X_balanced, y_balanced)
    
    # 5. Evaluate model
    scores = evaluate_model(best_model, X, y)
    print("\nModel Performance:")
    for metric, score in scores.items():
        print(f"{metric.capitalize()}: {score['mean']:.3f} (+/- {score['std']:.3f})")
    
    # 6. Save model artifacts
    model_data = {
        'model': best_model,
        'feature_names': list(X.columns),
        'scaler': scaler,
        'last_training_date': datetime.now().isoformat()
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

def load_model():
    """Load the trained model and associated artifacts"""
    try:
        model_data = joblib.load(MODEL_PATH)
        required_keys = ['model', 'feature_names', 'scaler']
        
        # Validate model data
        if not all(key in model_data for key in required_keys):
            raise KeyError("Missing required model artifacts")
        
        return model_data
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading model: {str(e)}")
        print("Training new model...")
        train_and_save(tune_hyperparams=True)
        return joblib.load(MODEL_PATH)

def predict_spoilage(model_data, product, days_old, temperature, humidity,
                    category=None, storage_type='Ambient', store_location=None):
    """Make predictions with enhanced feature engineering"""
    
    # Create a single sample dataframe
    sample = pd.DataFrame({
        'product': [product],
        'category': [category if category else 'Unknown'],
        'storage_type': [storage_type],
        'store_location': [store_location if store_location else 'Unknown'],
        'temperature': [temperature],
        'humidity': [humidity],
        'stock_date': [pd.Timestamp.now() - pd.Timedelta(days=days_old)],
        'expiry_date': [pd.Timestamp.now() + pd.Timedelta(days=7)]  # Default shelf life
    })
    
    # Engineer features
    sample = engineer_features(sample)
    X, _ = create_feature_matrix(sample)
    
    # Ensure all features from training are present
    missing_cols = set(model_data['feature_names']) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
        
    # Reorder columns to match training data
    X = X[model_data['feature_names']]
    
    # Predict probability of spoilage
    return model_data['model'].predict_proba(X)[0][1]

if __name__ == "__main__":
    train_and_save(tune_hyperparams=True)

