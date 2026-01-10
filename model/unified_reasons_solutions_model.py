#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Reasons & Solutions Model
Mô hình thống nhất cho 7 loại reasons & solutions:
1. Teaching Methods (PPGD)
2. Evaluation Methods (PPDG)
3. Student Conduct (Điểm rèn luyện)
4. Academic Midterm (Điểm giữa kỳ)
5. CLO Attendance (Chuyên cần CLO)
6. Self-Study (Tự học)
7. Attendance (Điểm danh) - từ file Excel
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
        'self_study': 'dulieu/student_selfstudy_reason_solution_dataset_v1_5000.csv',
        'attendance': 'dulieu/Dữ liệu điểm danh Khoa FIRA.xlsx'  # Dataset thứ 7: File Excel điểm danh
    }
    
    # Mô tả từng dataset
    DATASET_DESCRIPTIONS = {
        'teaching_methods': 'Phương pháp giảng dạy (PPGD)',
        'evaluation_methods': 'Phương pháp đánh giá (PPDG)',
        'student_conduct': 'Điểm rèn luyện',
        'academic_midterm': 'Điểm giữa kỳ',
        'clo_attendance': 'Chuyên cần CLO',
        'self_study': 'Tự học',
        'attendance': 'Điểm danh'
    }
    
    def __init__(self):
        """Khởi tạo model"""
        self.datasets = {}
        self.models = {}
        self.label_encoders = {}
        self.severity_encoders = {}
        
    def load_all_datasets(self):
        """Tải tất cả các datasets (hỗ trợ cả CSV và Excel)"""
        print("=" * 80)
        print("ĐANG TẢI TẤT CẢ CÁC DATASETS")
        print("=" * 80)
        
        for key, filepath in self.DATASET_FILES.items():
            try:
                import os
                if not os.path.exists(filepath):
                    print(f"⚠️  {self.DATASET_DESCRIPTIONS[key]:30} | File không tồn tại: {filepath}")
                    continue
                
                # Hỗ trợ cả CSV và Excel
                if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                    # Load file Excel và xử lý thành dataset format
                    df = self._process_excel_attendance(filepath)
                    if df is not None:
                        self.datasets[key] = df
                        print(f"✅ {self.DATASET_DESCRIPTIONS[key]:30} | {len(df):5} bản ghi | {filepath} (Excel)")
                else:
                    # Load file CSV
                    df = pd.read_csv(filepath, encoding='utf-8')
                    self.datasets[key] = df
                    print(f"✅ {self.DATASET_DESCRIPTIONS[key]:30} | {len(df):5} bản ghi | {filepath}")
            except Exception as e:
                print(f"❌ {self.DATASET_DESCRIPTIONS[key]:30} | Lỗi: {e}")
                import traceback
                traceback.print_exc()
                
        print(f"\n📊 Tổng số datasets đã load: {len(self.datasets)}/{len(self.DATASET_FILES)}")
        if len(self.datasets) < len(self.DATASET_FILES):
            missing = set(self.DATASET_FILES.keys()) - set(self.datasets.keys())
            print(f"⚠️  Thiếu {len(missing)} dataset(s): {', '.join(missing)}")
        return len(self.datasets) > 0
    
    def _process_excel_attendance(self, filepath):
        """Xử lý file Excel điểm danh thành format dataset"""
        try:
            # Load file Excel
            df_raw = pd.read_excel(filepath)
            
            # Tính điểm chuyên cần dựa trên điểm danh
            # Điểm danh: "Sớm" = tốt, "Muộn" = trung bình, "Vắng" = kém
            attendance_score_map = {
                'Sớm': 1.0,      # Tốt nhất
                'Muộn': 0.6,     # Trung bình
                'Vắng': 0.2,     # Kém
                'Có mặt': 1.0,
                'Vắng mặt': 0.2
            }
            
            # Tính điểm chuyên cần cho từng sinh viên-môn học
            df_processed = df_raw.groupby(['MSSV', 'Mã môn học']).agg({
                'Điểm danh': lambda x: x.map(attendance_score_map).fillna(0.5).mean(),  # Điểm trung bình
                'Mã giảng viên': 'first',
                'Tên môn học': 'first'
            }).reset_index()
            
            df_processed.columns = ['MSSV', 'Subject_ID', 'attendance_score', 'Lecturer_ID', 'Subject_Name']
            
            # Normalize điểm về 0-1
            df_processed['attendance_score_normalized'] = df_processed['attendance_score']
            
            # Tạo severity_level dựa trên điểm
            def get_severity(score):
                if score >= 0.86:
                    return 'Tốt'
                elif score >= 0.61:
                    return 'Thấp'
                elif score >= 0.26:
                    return 'Trung bình'
                else:
                    return 'Cao'
            
            df_processed['severity_level'] = df_processed['attendance_score_normalized'].apply(get_severity)
            
            # Tạo reason_text và solution_text dựa trên điểm
            def generate_reason(score, subject_name):
                if score < 0.26:
                    return f"[{subject_name}] Sinh viên vắng mặt nhiều, chuyên cần kém"
                elif score < 0.61:
                    return f"[{subject_name}] Sinh viên đi học không đều, có nhiều buổi muộn"
                elif score < 0.86:
                    return f"[{subject_name}] Chuyên cần khá tốt, cần duy trì"
                else:
                    return f"[{subject_name}] Chuyên cần tốt, đi học đầy đủ"
            
            def generate_solution(score):
                if score < 0.26:
                    return "Nhắc nhở và cảnh báo về tình trạng vắng mặt, yêu cầu cải thiện chuyên cần"
                elif score < 0.61:
                    return "Tăng cường kiểm tra điểm danh, nhắc nhở sinh viên đi học đúng giờ"
                elif score < 0.86:
                    return "Duy trì và khuyến khích sinh viên tiếp tục đi học đầy đủ"
                else:
                    return "Ghi nhận và khen thưởng sinh viên có chuyên cần tốt"
            
            df_processed['reason_text'] = df_processed.apply(
                lambda row: generate_reason(row['attendance_score_normalized'], row['Subject_Name']), axis=1
            )
            df_processed['solution_text'] = df_processed['attendance_score_normalized'].apply(generate_solution)
            
            # Đổi tên cột để phù hợp với format dataset
            df_processed['id'] = range(1, len(df_processed) + 1)
            df_processed['attendance_pred'] = df_processed['attendance_score_normalized']
            
            # Chọn các cột cần thiết (giống format các dataset khác)
            df_final = df_processed[['id', 'attendance_pred', 'severity_level', 'reason_text', 'solution_text']].copy()
            
            # Giới hạn số lượng mẫu (nếu quá nhiều)
            if len(df_final) > 5000:
                df_final = df_final.sample(n=5000, random_state=42).reset_index(drop=True)
                df_final['id'] = range(1, len(df_final) + 1)
            
            return df_final
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file Excel điểm danh: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            if 'self_study_score' in df.columns:
                feature_cols.append('self_study_score')
        elif dataset_key == 'attendance':
            # Dataset thứ 7: Điểm danh (từ Excel)
            if 'attendance_pred' in df.columns:
                feature_cols.append('attendance_pred')
            elif 'attendance_score' in df.columns:
                feature_cols.append('attendance_score')
            elif 'attendance_score_normalized' in df.columns:
                feature_cols.append('attendance_score_normalized')
        
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
        
        # Train Random Forest - Tăng tham số để đạt độ chính xác cao hơn
        rf_model = RandomForestClassifier(
            n_estimators=1000,  # Tăng từ 200 lên 1000 - nhiều cây hơn = chính xác hơn
            max_depth=25,        # Tăng từ 15 lên 25 - cây sâu hơn
            min_samples_split=2,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1  # Sử dụng tất cả CPU cores để train nhanh hơn
        )
        
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        
        print(f"\n✅ Random Forest Accuracy: {rf_score:.4f}")
        
        # Train Gradient Boosting - Tăng tham số để đạt độ chính xác cao hơn
        gb_model = GradientBoostingClassifier(
            n_estimators=500,   # Tăng từ 100 lên 500 - nhiều boosting stages hơn
            max_depth=12,        # Tăng từ 10 lên 12 - cây sâu hơn
            learning_rate=0.03,  # Giảm từ 0.1 xuống 0.03 - học chậm hơn nhưng chính xác hơn
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
    
    def _get_severity_from_score(self, score_normalized: float, dataset_key: str = 'clo_attendance') -> str:
        """
        Xác định severity dựa trên điểm (rule-based để đảm bảo chính xác)
        Dựa trên phân tích dataset clo_attendance:
        - Cao: 0.000 - 0.260 (điểm rất thấp, vấn đề nghiêm trọng)
        - Trung bình: 0.260 - 0.610 (điểm trung bình)
        - Thấp: 0.611 - 0.860 (điểm tốt, severity thấp = vấn đề ít)
        - Tốt: 0.860 - 1.000 (điểm rất tốt)
        
        Áp dụng cùng logic cho tất cả datasets
        """
        if score_normalized < 0.26:
            return 'Cao'
        elif score_normalized < 0.61:
            return 'Trung bình'
        elif score_normalized < 0.86:
            return 'Thấp'
        else:
            return 'Tốt'
    
    def _get_score_column_name(self, dataset_key: str) -> str:
        """Lấy tên cột điểm cho từng dataset"""
        score_columns = {
            'clo_attendance': 'clo_score_pred',
            'teaching_methods': 'teaching_method_pred',
            'evaluation_methods': 'evaluation_method_pred',
            'student_conduct': 'conduct_score_pred',
            'academic_midterm': 'midterm_score',
            'self_study': 'self_study_score',
            'attendance': 'attendance_pred'  # Dataset thứ 7
        }
        return score_columns.get(dataset_key, 'clo_score_pred')
    
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
        
        # Lấy điểm từ features (features[0] là điểm normalized)
        score_normalized = features[0] if len(features) > 0 else 0.5
        
        # QUAN TRỌNG: Luôn dùng rule-based để xác định severity (chặt chẽ, đảm bảo đúng)
        severity_label = self._get_severity_from_score(score_normalized, dataset_key)
        
        # Tính confidence dựa trên khoảng cách đến ngưỡng
        # Điểm càng gần trung tâm của severity range thì confidence càng cao
        severity_ranges = {
            'Cao': (0.0, 0.26),
            'Trung bình': (0.26, 0.61),
            'Thấp': (0.61, 0.86),
            'Tốt': (0.86, 1.0)
        }
        
        if severity_label in severity_ranges:
            min_score, max_score = severity_ranges[severity_label]
            range_center = (min_score + max_score) / 2
            # Confidence cao nếu điểm gần trung tâm của range
            distance_from_center = abs(score_normalized - range_center)
            max_distance = (max_score - min_score) / 2
            severity_confidence = max(0.5, 1.0 - (distance_from_center / max_distance))
        else:
            severity_confidence = 0.7
        
        # Dự đoán severity bằng model (chỉ để tham khảo, không dùng)
        # Tạo features đầy đủ
        full_features = list(features)
        while len(full_features) < len(feature_names):
            full_features.append(100)  # Giá trị mặc định cho length
        
        try:
            X = np.array([full_features]).reshape(1, -1)
            severity_pred = model.predict(X)[0]
            severity_proba = model.predict_proba(X)[0]
            severity_model = self.severity_encoders[dataset_key].inverse_transform([severity_pred])[0]
            
            # Nếu model predict khác với rule-based, cảnh báo và giảm confidence
            if severity_model != severity_label:
                # Model predict sai, giảm confidence và ưu tiên rule-based
                severity_confidence = min(severity_confidence, 0.6)
        except:
            # Nếu model chưa train hoặc lỗi, chỉ dùng rule-based
            pass
        
        # QUAN TRỌNG: Lọc reasons/solutions theo severity ĐÚNG và điểm GẦN với input
        filtered_df = df[df['severity_level'] == severity_label].copy()
        
        if len(filtered_df) == 0:
            # Nếu không tìm thấy, tìm severity gần nhất
            severity_ranges = {
                'Cao': (0.0, 0.26),
                'Trung bình': (0.26, 0.61),
                'Thấp': (0.61, 0.86),
                'Tốt': (0.86, 1.0)
            }
            # Tìm severity có range gần với điểm nhất
            min_dist = float('inf')
            closest_severity = severity_label
            for sev, (min_s, max_s) in severity_ranges.items():
                if min_s <= score_normalized <= max_s:
                    closest_severity = sev
                    break
                dist = min(abs(score_normalized - min_s), abs(score_normalized - max_s))
                if dist < min_dist:
                    min_dist = dist
                    closest_severity = sev
            filtered_df = df[df['severity_level'] == closest_severity].copy()
            severity_label = closest_severity  # Cập nhật severity label
        
        # Tìm cột điểm trong dataset
        score_col = self._get_score_column_name(dataset_key)
        if score_col not in filtered_df.columns:
            # Fallback: thử các cột điểm phổ biến
            for col in ['clo_score_pred', 'teaching_method_pred', 'conduct_score_pred', 'midterm_score']:
                if col in filtered_df.columns:
                    score_col = col
                    break
        
        # QUAN TRỌNG: Ưu tiên những reasons/solutions có điểm GẦN với điểm input
        if score_col in filtered_df.columns:
            # Tính khoảng cách điểm
            filtered_df['score_diff'] = abs(filtered_df[score_col] - score_normalized)
            # Sắp xếp theo điểm gần nhất
            filtered_df = filtered_df.sort_values('score_diff')
        else:
            # Nếu không có cột điểm, sắp xếp ngẫu nhiên
            filtered_df = filtered_df.sample(frac=1, random_state=42)
        
        # Lấy top_k reasons & solutions (ưu tiên điểm gần nhất)
        if len(filtered_df) > top_k:
            samples = filtered_df.head(top_k)
        else:
            samples = filtered_df.head(min(len(filtered_df), top_k))
        
        results = []
        for idx, row in samples.iterrows():
            # Tính độ khớp điểm nếu có cột điểm
            score_match = 1.0
            if score_col in row.index:
                score_match = float(1.0 - min(abs(row[score_col] - score_normalized), 1.0))
            
            results.append({
                'reason': row['reason_text'],
                'solution': row['solution_text'],
                'severity': severity_label,
                'confidence': float(severity_confidence),
                'score_match': score_match  # Độ khớp điểm (0-1)
            })
        
        return {
            'dataset': self.DATASET_DESCRIPTIONS[dataset_key],
            'severity_level': severity_label,
            'severity_confidence': float(severity_confidence),
            'score_normalized': float(score_normalized),
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

