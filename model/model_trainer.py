import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .config import RF_PARAMS, GB_PARAMS
from .utils import safe_float
from sklearn.linear_model import LogisticRegression
import json

class ModelTrainer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.df = data_loader.df
        self.model = None
        self.X = None
        self.y = None

    def prepare_data(self):
        """Prepare X and y for training"""
        # Filter features that exist in the dataframe
        available_features = [col for col in self.data_loader.feature_names if col in self.df.columns]
        missing_features = [col for col in self.data_loader.feature_names if col not in self.df.columns]
        
        if missing_features:
            print(f"Warning: Missing features in model training: {missing_features}")
        
        self.X = self.df[available_features]
        self.y = self.df['passed']
        
        print(f"Features used: {len(available_features)}")
        print(f"Feature names: {available_features}")
        
        # Update feature names to only include available ones
        self.data_loader.feature_names = available_features

    def optimize_hyperparameters(self, X, y, n_iter=20, cv=3, random_state=42):
        """Tối ưu tham số cho Logistic Regression, Random Forest và Gradient Boosting bằng RandomizedSearchCV. Ghi kết quả vào model_stats.txt"""
        print("\n=== TỐI ƯU THAM SỐ (RandomizedSearchCV) ===")
        lr_param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [500, 1000]
        }
        rf_param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['auto', 'sqrt', 'log2', None],
        }
        gb_param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 1.0],
        }
        lr = LogisticRegression(random_state=random_state)
        rf = RandomForestClassifier(random_state=random_state)
        gb = GradientBoostingClassifier(random_state=random_state)
        lr_search = RandomizedSearchCV(lr, lr_param_grid, n_iter=n_iter, cv=cv, scoring='accuracy', random_state=random_state, n_jobs=-1, verbose=1)
        rf_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=n_iter, cv=cv, scoring='accuracy', random_state=random_state, n_jobs=-1, verbose=1)
        gb_search = RandomizedSearchCV(gb, gb_param_grid, n_iter=n_iter, cv=cv, scoring='accuracy', random_state=random_state, n_jobs=-1, verbose=1)
        print("Tối ưu Logistic Regression...")
        lr_search.fit(X, y)
        print(f"Best LR params: {lr_search.best_params_}, best score: {lr_search.best_score_:.4f}")
        print("Tối ưu Random Forest...")
        rf_search.fit(X, y)
        print(f"Best RF params: {rf_search.best_params_}, best score: {rf_search.best_score_:.4f}")
        print("Tối ưu Gradient Boosting...")
        gb_search.fit(X, y)
        print(f"Best GB params: {gb_search.best_params_}, best score: {gb_search.best_score_:.4f}")
        # Ghi kết quả vào file model_stats.txt
        stats = {
            'LogisticRegression': {'best_params': lr_search.best_params_, 'best_score': lr_search.best_score_},
            'RandomForest': {'best_params': rf_search.best_params_, 'best_score': rf_search.best_score_},
            'GradientBoosting': {'best_params': gb_search.best_params_, 'best_score': gb_search.best_score_}
        }
        with open('model_stats.txt', 'w', encoding='utf-8') as f:
            f.write(json.dumps(stats, indent=2, ensure_ascii=False))
        return lr_search.best_params_, rf_search.best_params_, gb_search.best_params_

    def train_models(self, optimize_params=False):
        """Train the ensemble model"""
        print("Training models...")
        
        def safe_float(x):
            try:
                if pd.isna(x):
                    return 0.0
                if isinstance(x, str):
                    if x.upper() == 'VT':
                        return 0.0
                    x = x.replace(',', '.')
                return float(x)
            except (ValueError, TypeError):
                return 0.0
        
        # Convert all features to numeric
        for col in self.X.columns:
            self.X[col] = self.X[col].apply(safe_float)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Tối ưu tham số nếu được yêu cầu
        lr_params = {'max_iter': 1000, 'random_state': 42}
        rf_params, gb_params = RF_PARAMS, GB_PARAMS
        if optimize_params:
            lr_params, rf_params, gb_params = self.optimize_hyperparameters(X_train, y_train)
        # Initialize models
        lr_model = LogisticRegression(**lr_params)
        rf_model = RandomForestClassifier(**rf_params)
        gb_model = GradientBoostingClassifier(**gb_params)
        # Train individual models
        print("Training Logistic Regression (baseline)...")
        lr_model.fit(X_train, y_train)
        lr_score = lr_model.score(X_test, y_test)
        print(f"Logistic Regression accuracy: {lr_score:.4f}")
        print("Training Random Forest...")
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        print(f"Random Forest accuracy: {rf_score:.4f}")
        print("Training Gradient Boosting...")
        gb_model.fit(X_train, y_train)
        gb_score = gb_model.score(X_test, y_test)
        print(f"Gradient Boosting accuracy: {gb_score:.4f}")
        # Create optimized ensemble model (RF + GB only for best performance)
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft'
        )
        # Train ensemble
        print("Training ensemble model...")
        self.model.fit(X_train, y_train)
        ensemble_score = self.model.score(X_test, y_test)
        print(f"Ensemble accuracy: {ensemble_score:.4f}")
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        # Feature importance
        if hasattr(rf_model, 'feature_importances_'):
            feature_importance = rf_model.feature_importances_
            feature_names = self.data_loader.feature_names
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            print("\nTop 10 most important features:")
            print(importance_df.head(10))
        return {
            'lr_score': lr_score,
            'rf_score': rf_score,
            'gb_score': gb_score,
            'ensemble_score': ensemble_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            print("No model trained yet. Run train_models() first.")
            return
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        } 