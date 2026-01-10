import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# Turn off unnecessary warnings
warnings.filterwarnings('ignore')

# === HYPERPARAMETERS FOR REGRESSION MODELS ===
# Tối ưu cho độ chính xác cao - tăng số lượng trees/iterations
RF_PARAMS = {
    'n_estimators': 1000,    # Tăng từ 300 lên 1000 - nhiều cây hơn = chính xác hơn
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': 1,    # Minimum samples per leaf (optimized)
    'max_features': 'sqrt',   # Number of features to consider for best split (changed from 'auto' for sklearn compatibility)
    'max_depth': 25,          # Tăng từ 20 lên 25 - cây sâu hơn
    'class_weight': 'balanced_subsample',  # Handle class imbalance
    'random_state': 42
}

GB_PARAMS = {
    'subsample': 0.8,         # Fraction of samples used for fitting (optimized)
    'n_estimators': 500,      # Tăng từ 100 lên 500 - nhiều boosting stages hơn
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': 2,    # Minimum samples per leaf
    'max_depth': 12,          # Tăng từ 10 lên 12 - cây sâu hơn
    'learning_rate': 0.03,     # Giảm từ 0.05 xuống 0.03 - học chậm hơn nhưng chính xác hơn (kết hợp với n_estimators cao)
    'random_state': 42
}

# Subject replacement mapping

LR_PARAMS = {
    'C': 0.1,                    # Inverse of regularization strength (optimized)
    'penalty': 'l2',       # Regularization penalty (optimized)
    'solver': 'liblinear',        # Algorithm for optimization (optimized)
    'max_iter': 5000,      # Tăng từ 1000 lên 5000 - nhiều iterations hơn
    'random_state': 42
}

ET_PARAMS = {
    'n_estimators': 500,      # Tăng từ 200 lên 500 - nhiều cây hơn
    'max_depth': 20,          # Tăng từ 15 lên 20 - cây sâu hơn
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': 1,    # Minimum samples per leaf (optimized)
    'max_features': 'sqrt',   # Number of features to consider for best split (changed from 'auto' for sklearn compatibility)
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

# File paths - Updated to use dulieu/ folder
# PPGDfull: Từ 51 môn lên 91 môn (cập nhật với dữ liệu điểm danh mới)
DATA_FILES = {
    'main_data': 'dulieu/DiemTong.xlsx',
    'teaching_methods': 'dulieu/PPGDfull.xlsx',  # Đã thay PPGD.xlsx bằng PPGDfull.xlsx (91 môn)
    'assessment_methods': 'dulieu/PPDGfull.xlsx',
    'attendance': 'dulieu/Dữ liệu điểm danh Khoa FIRA.xlsx',  # File điểm danh mới
    'demographic': 'dulieu/nhankhau.xlsx',
    'conduct': 'dulieu/diemrenluyen.xlsx',
    'self_study': 'dulieu/tuhoc.xlsx'
} 