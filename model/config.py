import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# Turn off unnecessary warnings
warnings.filterwarnings('ignore')

# === HYPERPARAMETERS FOR REGRESSION MODELS ===
# Optimized parameters based on comprehensive analysis
RF_PARAMS = {
    'n_estimators': 300,      # Number of trees in Random Forest (optimized)
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': 1,    # Minimum samples per leaf (optimized)
    'max_features': 'auto',   # Number of features to consider for best split
    'max_depth': 20,          # Maximum depth of each tree (optimized)
    'class_weight': 'balanced_subsample',  # Handle class imbalance
    'random_state': 42
}

GB_PARAMS = {
    'subsample': 0.8,         # Fraction of samples used for fitting (optimized)
    'n_estimators': 100,      # Number of boosting stages (optimized)
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': 2,    # Minimum samples per leaf
    'max_depth': 10,          # Maximum depth of each tree (optimized)
    'learning_rate': 0.05,    # Learning rate shrinks the contribution of each tree (optimized)
    'random_state': 42
}

# Subject replacement mapping

LR_PARAMS = {
    'C': 0.1,                    # Inverse of regularization strength (optimized)
    'penalty': 'l2',       # Regularization penalty (optimized)
    'solver': 'liblinear',        # Algorithm for optimization (optimized)
    'max_iter': 1000,      # Maximum iterations (optimized)
    'random_state': 42
}

ET_PARAMS = {
    'n_estimators': 200,      # Number of trees in Extra Trees (optimized)
    'max_depth': 15,          # Maximum depth of each tree (optimized)
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': 1,    # Minimum samples per leaf (optimized)
    'max_features': 'auto',   # Number of features to consider for best split
    'random_state': 42
}

# === OPTIMIZED ENSEMBLE CONFIGURATION ===
BEST_ENSEMBLE_CONFIG = {
    'name': 'Voting_Soft_Top3',
    'voting': 'soft',
    'estimators': ['GradientBoosting', 'RandomForest', 'ExtraTrees']
}

# === OPTIMIZATION RESULTS SUMMARY ===
OPTIMIZATION_SUMMARY = {
    'best_individual_model': 'GradientBoosting',
    'best_ensemble_model': 'Voting_Soft_Top3',
    'optimization_timestamp': '2025-07-28',
    'performance_metrics': {
        'best_f1_score': 0.9502,
        'best_accuracy': 0.9205,
        'best_roc_auc': 0.9567
    }
}

SUBJECT_REPLACE = {
    'POL0072': 'PLO0072',
    'POL0043': 'PLO0043',
    'POL0032': 'PLO0032',
    'POL0052': 'PLO0052'
}

# File paths - Updated to use dulieu/ folder
DATA_FILES = {
    'main_data': 'dulieu/DiemTong.xlsx',
    'teaching_methods': 'dulieu/PPGD.xlsx',
    'assessment_methods': 'dulieu/PPDG.xlsx',
    'demographic': 'dulieu/nhankhau.xlsx',
    'conduct': 'dulieu/diemrenluyen.xlsx',
    'self_study': 'dulieu/tuhoc.xlsx'
} 