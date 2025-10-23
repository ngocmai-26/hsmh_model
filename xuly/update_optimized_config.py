#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Optimized Config
Cập nhật config.py với các tham số tối ưu từ kết quả tối ưu hóa
"""

import json
import re

def load_optimized_parameters():
    """Load optimized parameters from JSON file"""
    try:
        with open('fast_optimized_parameters.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: fast_optimized_parameters.json not found!")
        return None

def update_config_file():
    """Update config.py with optimized parameters"""
    print("=== UPDATING CONFIG WITH OPTIMIZED PARAMETERS ===")
    
    # Load optimized parameters
    optimized_params = load_optimized_parameters()
    if not optimized_params:
        return
    
    # Read current config.py
    try:
        with open('config.py', 'r', encoding='utf-8') as f:
            config_content = f.read()
    except FileNotFoundError:
        print("Error: config.py not found!")
        return
    
    print(f"Best individual model: {optimized_params['best_individual_model']}")
    print(f"Best ensemble model: {optimized_params['best_ensemble_model']}")
    
    # Update RF_PARAMS
    rf_params = optimized_params['individual_models']['RandomForest']
    rf_params_str = f"""RF_PARAMS = {{
    'n_estimators': {rf_params['n_estimators']},      # Number of trees in Random Forest (optimized)
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': {rf_params['min_samples_leaf']},    # Minimum samples per leaf (optimized)
    'max_features': '{rf_params['max_features']}',   # Number of features to consider for best split
    'max_depth': {rf_params['max_depth']},          # Maximum depth of each tree (optimized)
    'class_weight': 'balanced_subsample',  # Handle class imbalance
    'random_state': 42
}}"""
    
    # Update GB_PARAMS
    gb_params = optimized_params['individual_models']['GradientBoosting']
    gb_params_str = f"""GB_PARAMS = {{
    'subsample': {gb_params['subsample']},         # Fraction of samples used for fitting (optimized)
    'n_estimators': {gb_params['n_estimators']},      # Number of boosting stages (optimized)
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': 2,    # Minimum samples per leaf
    'max_depth': {gb_params['max_depth']},          # Maximum depth of each tree (optimized)
    'learning_rate': {gb_params['learning_rate']},    # Learning rate shrinks the contribution of each tree (optimized)
    'random_state': 42
}}"""
    
    # Add new optimized parameters
    lr_params = optimized_params['individual_models']['LogisticRegression']
    et_params = optimized_params['individual_models']['ExtraTrees']
    
    lr_params_str = f"""LR_PARAMS = {{
    'C': {lr_params['C']},                    # Inverse of regularization strength (optimized)
    'penalty': '{lr_params['penalty']}',       # Regularization penalty (optimized)
    'solver': '{lr_params['solver']}',        # Algorithm for optimization (optimized)
    'max_iter': {lr_params['max_iter']},      # Maximum iterations (optimized)
    'random_state': 42
}}"""
    
    et_params_str = f"""ET_PARAMS = {{
    'n_estimators': {et_params['n_estimators']},      # Number of trees in Extra Trees (optimized)
    'max_depth': {et_params['max_depth']},          # Maximum depth of each tree (optimized)
    'min_samples_split': 2,   # Minimum samples required to split
    'min_samples_leaf': {et_params['min_samples_leaf']},    # Minimum samples per leaf (optimized)
    'max_features': 'auto',   # Number of features to consider for best split
    'random_state': 42
}}"""
    
    # Add ensemble configuration
    ensemble_config_str = f"""# === OPTIMIZED ENSEMBLE CONFIGURATION ===
BEST_ENSEMBLE_CONFIG = {{
    'name': '{optimized_params['best_ensemble_model']}',
    'voting': 'soft',
    'estimators': ['GradientBoosting', 'RandomForest', 'ExtraTrees']
}}

# === OPTIMIZATION RESULTS SUMMARY ===
OPTIMIZATION_SUMMARY = {{
    'best_individual_model': '{optimized_params['best_individual_model']}',
    'best_ensemble_model': '{optimized_params['best_ensemble_model']}',
    'optimization_timestamp': '2025-07-28',
    'performance_metrics': {{
        'best_f1_score': 0.9502,
        'best_accuracy': 0.9205,
        'best_roc_auc': 0.9567
    }}
}}"""
    
    # Replace existing parameter definitions
    # Replace RF_PARAMS
    rf_pattern = r'RF_PARAMS\s*=\s*\{[^}]*\}'
    config_content = re.sub(rf_pattern, rf_params_str, config_content, flags=re.DOTALL)
    
    # Replace GB_PARAMS
    gb_pattern = r'GB_PARAMS\s*=\s*\{[^}]*\}'
    config_content = re.sub(gb_pattern, gb_params_str, config_content, flags=re.DOTALL)
    
    # Add new parameter definitions after existing ones
    # Find the end of existing parameters
    if 'ET_PARAMS' not in config_content:
        # Add new parameters before SUBJECT_REPLACE
        subject_replace_pos = config_content.find('SUBJECT_REPLACE')
        if subject_replace_pos != -1:
            new_params = f"\n{lr_params_str}\n\n{et_params_str}\n\n{ensemble_config_str}\n\n"
            config_content = config_content[:subject_replace_pos] + new_params + config_content[subject_replace_pos:]
    
    # Write updated config
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✓ Config.py updated with optimized parameters!")
    print("\nUpdated parameters:")
    print(f"  - RandomForest: n_estimators={rf_params['n_estimators']}, max_depth={rf_params['max_depth']}")
    print(f"  - GradientBoosting: n_estimators={gb_params['n_estimators']}, learning_rate={gb_params['learning_rate']}")
    print(f"  - LogisticRegression: C={lr_params['C']}, solver={lr_params['solver']}")
    print(f"  - ExtraTrees: n_estimators={et_params['n_estimators']}, max_depth={et_params['max_depth']}")
    print(f"  - Best ensemble: {optimized_params['best_ensemble_model']}")

def create_optimization_summary():
    """Create a summary of optimization results"""
    print("\n=== OPTIMIZATION SUMMARY ===")
    
    try:
        with open('fast_optimization_report.json', 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        best_individual = report['best_individual_model']
        best_ensemble = report['best_ensemble_model']
        
        print(f"📊 OPTIMIZATION RESULTS:")
        print(f"   • Best Individual Model: {best_individual['name']}")
        print(f"     - F1-Score: {best_individual['metrics']['f1_score']:.4f}")
        print(f"     - Accuracy: {best_individual['metrics']['accuracy']:.4f}")
        print(f"     - ROC AUC: {best_individual['metrics']['roc_auc']:.4f}")
        
        print(f"\n   • Best Ensemble Model: {best_ensemble['name']}")
        print(f"     - F1-Score: {best_ensemble['metrics']['f1_score']:.4f}")
        print(f"     - Accuracy: {best_ensemble['metrics']['accuracy']:.4f}")
        print(f"     - ROC AUC: {best_ensemble['metrics']['roc_auc']:.4f}")
        
        print(f"\n   • Data Information:")
        print(f"     - Total samples: {report['data_info']['total_samples']:,}")
        print(f"     - Features used: {report['data_info']['features_used']}")
        print(f"     - Pass rate: {report['data_info']['pass_rate']:.2%}")
        
        # Show top 3 models
        all_models = report['all_models_performance']
        sorted_models = sorted(all_models.items(), 
                             key=lambda x: x[1]['test_metrics']['f1_score'], 
                             reverse=True)
        
        print(f"\n   • Top 3 Individual Models:")
        for i, (name, results) in enumerate(sorted_models[:3], 1):
            f1 = results['test_metrics']['f1_score']
            acc = results['test_metrics']['accuracy']
            print(f"     {i}. {name}: F1={f1:.4f}, Accuracy={acc:.4f}")
        
        # Show recommendations
        print(f"\n   • Key Recommendations:")
        for rec in report['recommendations'][:3]:
            print(f"     • {rec['recommendation']}")
            
    except FileNotFoundError:
        print("Error: fast_optimization_report.json not found!")

def main():
    """Main function"""
    print("=== MODEL OPTIMIZATION CONFIG UPDATER ===")
    
    # Update config with optimized parameters
    update_config_file()
    
    # Create optimization summary
    create_optimization_summary()
    
    print("\n✓ Optimization and config update completed!")
    print("You can now use the optimized parameters in your models.")

if __name__ == "__main__":
    main() 