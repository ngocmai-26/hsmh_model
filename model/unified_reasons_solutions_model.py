#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Reasons & Solutions Model
Mô hình thống nhất cho 6 loại reasons & solutions:
1. Teaching Methods (PPGD)
2. Evaluation Methods (PPDG)
3. Student Conduct (Điểm rèn luyện)
4. Academic Midterm (Điểm giữa kỳ)
5. CLO Attendance (Chuyên cần CLO)
6. Self-Study (Tự học)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class UnifiedReasonsSolutionsModel:
    """Mô hình thống nhất cho tất cả các loại reasons & solutions"""
    
    # Định nghĩa các file dữ liệu
    DATASET_FILES = {
        'teaching_methods': 'dulieu/teaching_methods_reason_solution_dataset_v2_5000.csv',
        'evaluation_methods': 'dulieu/evaluation_methods_reason_solution_dataset_v2_5000.csv',
        'student_conduct': 'dulieu/student_conduct_reason_solution_dataset_v1_5000.csv',
        'academic_midterm': 'dulieu/academic_midterm_reason_solution_dataset_5000.csv',
        'clo_attendance': 'dulieu/clo_attendance_reason_solution_dataset_5000.csv',
        'self_study': 'dulieu/student_selfstudy_reason_solution_dataset_v1_5000.csv'
    }
    
    # Mô tả từng dataset
    DATASET_DESCRIPTIONS = {
        'teaching_methods': 'Phương pháp giảng dạy (PPGD)',
        'evaluation_methods': 'Phương pháp đánh giá (PPDG)',
        'student_conduct': 'Điểm rèn luyện',
        'academic_midterm': 'Điểm giữa kỳ',
        'clo_attendance': 'Chuyên cần CLO',
        'self_study': 'Tự học'
    }
    
    def __init__(self):
        """Khởi tạo model"""
        self.datasets = {}
        self.models = {}
        self.label_encoders = {}
        self.severity_encoders = {}
        
    def load_all_datasets(self):
        """Tải tất cả các datasets"""
        print("=" * 80)
        print("ĐANG TẢI TẤT CẢ CÁC DATASETS")
        print("=" * 80)
        
        for key, filepath in self.DATASET_FILES.items():
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                self.datasets[key] = df
                print(f"✅ {self.DATASET_DESCRIPTIONS[key]:30} | {len(df):5} bản ghi | {filepath}")
            except Exception as e:
                print(f"❌ {self.DATASET_DESCRIPTIONS[key]:30} | Lỗi: {e}")
                
        print(f"\n📊 Tổng số datasets: {len(self.datasets)}/{len(self.DATASET_FILES)}")
        return len(self.datasets) > 0
    
    def analyze_dataset_structure(self):
        """Phân tích cấu trúc của từng dataset"""
        print("\n" + "=" * 80)
        print("PHÂN TÍCH CẤU TRÚC CÁC DATASETS")
        print("=" * 80)
        
        for key, df in self.datasets.items():
            print(f"\n📋 {self.DATASET_DESCRIPTIONS[key]}")
            print(f"   Số cột: {len(df.columns)}")
            print(f"   Các cột: {', '.join(df.columns.tolist())}")
            
            # Phân tích severity_level
            if 'severity_level' in df.columns:
                severity_dist = df['severity_level'].value_counts()
                print(f"   Phân bố Severity:")
                for level, count in severity_dist.items():
                    print(f"      - {level}: {count} ({count/len(df)*100:.1f}%)")
    
    def prepare_training_data(self, dataset_key):
        """Chuẩn bị dữ liệu huấn luyện cho một dataset cụ thể"""
        if dataset_key not in self.datasets:
            print(f"❌ Dataset {dataset_key} không tồn tại!")
            return None, None, None, None
        
        df = self.datasets[dataset_key].copy()
        
        # Xác định cột features
        feature_cols = []
        target_col = 'severity_level'
        
        # Tùy chỉnh features theo từng dataset
        if dataset_key == 'teaching_methods':
            if 'teaching_method_pred' in df.columns:
                feature_cols.append('teaching_method_pred')
        elif dataset_key == 'evaluation_methods':
            if 'evaluation_method_pred' in df.columns:
                feature_cols.append('evaluation_method_pred')
        elif dataset_key == 'student_conduct':
            if 'conduct_score_pred' in df.columns:
                feature_cols.append('conduct_score_pred')
        elif dataset_key == 'academic_midterm':
            if 'midterm_score' in df.columns:
                feature_cols.append('midterm_score')
        elif dataset_key == 'clo_attendance':
            if 'clo_score_pred' in df.columns:
                feature_cols.append('clo_score_pred')
        elif dataset_key == 'self_study':
            # Thêm features cho self_study nếu có
            pass
        
        # Thêm text features
        if 'reason_text' in df.columns:
            df['reason_length'] = df['reason_text'].str.len()
            feature_cols.append('reason_length')
            
        if 'solution_text' in df.columns:
            df['solution_length'] = df['solution_text'].str.len()
            feature_cols.append('solution_length')
        
        # Encode target
        if target_col not in df.columns:
            print(f"❌ Không tìm thấy cột {target_col}")
            return None, None, None, None
            
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        self.severity_encoders[dataset_key] = le
        
        # Tạo X
        if not feature_cols:
            print(f"❌ Không có features cho dataset {dataset_key}")
            return None, None, None, None
            
        X = df[feature_cols].fillna(0)
        
        return X, y, df, feature_cols
    
    def train_model(self, dataset_key, test_size=0.2, random_state=42):
        """Huấn luyện mô hình cho một dataset cụ thể"""
        print(f"\n{'=' * 80}")
        print(f"HUẤN LUYỆN MÔ HÌNH: {self.DATASET_DESCRIPTIONS[dataset_key]}")
        print(f"{'=' * 80}")
        
        X, y, df, feature_cols = self.prepare_training_data(dataset_key)
        
        if X is None:
            print(f"❌ Không thể huấn luyện mô hình cho {dataset_key}")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"📊 Dữ liệu:")
        print(f"   - Train: {len(X_train)} mẫu")
        print(f"   - Test:  {len(X_test)} mẫu")
        print(f"   - Features: {feature_cols}")
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            random_state=random_state,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        
        print(f"\n✅ Random Forest Accuracy: {rf_score:.4f}")
        
        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=random_state
        )
        
        gb_model.fit(X_train, y_train)
        gb_score = gb_model.score(X_test, y_test)
        
        print(f"✅ Gradient Boosting Accuracy: {gb_score:.4f}")
        
        # Lưu mô hình tốt nhất
        if rf_score >= gb_score:
            self.models[dataset_key] = {
                'model': rf_model,
                'type': 'RandomForest',
                'accuracy': rf_score,
                'features': feature_cols,
                'data': df
            }
            print(f"\n🏆 Chọn Random Forest (accuracy: {rf_score:.4f})")
        else:
            self.models[dataset_key] = {
                'model': gb_model,
                'type': 'GradientBoosting',
                'accuracy': gb_score,
                'features': feature_cols,
                'data': df
            }
            print(f"\n🏆 Chọn Gradient Boosting (accuracy: {gb_score:.4f})")
        
        return self.models[dataset_key]
    
    def train_all_models(self):
        """Huấn luyện tất cả các mô hình"""
        print("\n" + "=" * 80)
        print("BẮT ĐẦU HUẤN LUYỆN TẤT CẢ CÁC MÔ HÌNH")
        print("=" * 80)
        
        results = {}
        for key in self.datasets.keys():
            result = self.train_model(key)
            if result:
                results[key] = result
        
        print("\n" + "=" * 80)
        print("KẾT QUẢ HUẤN LUYỆN")
        print("=" * 80)
        
        for key, result in results.items():
            print(f"{self.DATASET_DESCRIPTIONS[key]:30} | "
                  f"{result['type']:20} | "
                  f"Accuracy: {result['accuracy']:.4f}")
        
        return len(results)
    
    def predict_reason_solution(self, dataset_key, features, top_k=3):
        """Dự đoán reasons & solutions cho một dataset cụ thể"""
        if dataset_key not in self.models:
            return {
                'error': f'Model cho {dataset_key} chưa được huấn luyện'
            }
        
        model_info = self.models[dataset_key]
        model = model_info['model']
        df = model_info['data']
        feature_names = model_info['features']
        
        # Tạo features đầy đủ
        # features chỉ chứa score, cần thêm reason_length và solution_length
        full_features = list(features)
        
        # Thêm giá trị mặc định cho reason_length và solution_length nếu cần
        while len(full_features) < len(feature_names):
            full_features.append(100)  # Giá trị mặc định cho length
        
        # Dự đoán severity
        X = np.array([full_features]).reshape(1, -1)
        severity_pred = model.predict(X)[0]
        severity_proba = model.predict_proba(X)[0]
        
        # Decode severity
        severity_label = self.severity_encoders[dataset_key].inverse_transform([severity_pred])[0]
        severity_confidence = severity_proba[severity_pred]
        
        # Lấy top_k reasons & solutions
        filtered_df = df[df['severity_level'] == severity_label].copy()
        
        if len(filtered_df) == 0:
            filtered_df = df.copy()
        
        # Random sample top_k
        if len(filtered_df) > top_k:
            samples = filtered_df.sample(n=top_k, random_state=42)
        else:
            samples = filtered_df.sample(n=min(len(filtered_df), top_k), random_state=42)
        
        results = []
        for idx, row in samples.iterrows():
            results.append({
                'reason': row['reason_text'],
                'solution': row['solution_text'],
                'severity': severity_label,
                'confidence': float(severity_confidence)
            })
        
        return {
            'dataset': self.DATASET_DESCRIPTIONS[dataset_key],
            'severity_level': severity_label,
            'severity_confidence': float(severity_confidence),
            'results': results
        }
    
    def get_model_summary(self):
        """Lấy tóm tắt về các models"""
        summary = {
            'total_datasets': len(self.datasets),
            'total_models': len(self.models),
            'models': {}
        }
        
        for key, model_info in self.models.items():
            summary['models'][key] = {
                'description': self.DATASET_DESCRIPTIONS[key],
                'type': model_info['type'],
                'accuracy': model_info['accuracy'],
                'features': model_info['features']
            }
        
        return summary


def main():
    """Demo sử dụng model"""
    print("=" * 80)
    print("UNIFIED REASONS & SOLUTIONS MODEL - DEMO")
    print("=" * 80)
    
    # Khởi tạo model
    model = UnifiedReasonsSolutionsModel()
    
    # Load datasets
    if not model.load_all_datasets():
        print("❌ Không thể tải datasets!")
        return
    
    # Phân tích cấu trúc
    model.analyze_dataset_structure()
    
    # Huấn luyện tất cả models
    num_trained = model.train_all_models()
    print(f"\n✅ Đã huấn luyện thành công {num_trained} models!")
    
    # Demo prediction
    print("\n" + "=" * 80)
    print("DEMO DỰ ĐOÁN")
    print("=" * 80)
    
    # Ví dụ: Dự đoán teaching methods
    if 'teaching_methods' in model.models:
        result = model.predict_reason_solution('teaching_methods', [0.5], top_k=3)
        print(f"\n📚 {result['dataset']}")
        print(f"   Severity: {result['severity_level']} (confidence: {result['severity_confidence']:.3f})")
        for i, item in enumerate(result['results'], 1):
            print(f"\n   {i}. Nguyên nhân: {item['reason'][:100]}...")
            print(f"      Giải pháp: {item['solution'][:100]}...")
    
    # Hiển thị summary
    print("\n" + "=" * 80)
    print("TÓM TẮT MÔ HÌNH")
    print("=" * 80)
    summary = model.get_model_summary()
    print(f"Tổng số datasets: {summary['total_datasets']}")
    print(f"Tổng số models: {summary['total_models']}")


if __name__ == "__main__":
    main()

