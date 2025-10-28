import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class PPDGIntegration:
    """Tích hợp PPDG vào model AI để giải thích điểm CLO"""
    
    def __init__(self):
        """Initialize PPDG integration"""
        self.ppdg_mapping = {
            'EM 1': 'Đánh giá chuyên cần (Attendance And Punctuality Assessment)',
            'EM 2': 'Đánh giá bài tập cá nhân (Work Assignment Assessment)',
            'EM 3': 'Đánh giá thuyết trình (Oral Presentation Assessment)',
            'EM 4': 'Đánh giá làm việc nhóm (Teamwork Assessment)',
            'EM 5': 'Đánh giá tự học tại thư viện (Self-Study At The Library Assessment)',
            'EM 6': 'Kiểm tra viết (Written Exam)',
            'EM 7': 'Kiểm tra trắc nghiệm (Multiple Choice Exam)',
            'EM 8': 'Đánh giá báo cáo/tiểu luận (Written Report/Essay Assessment)',
            'EM 9': 'Đánh giá thực hành (Practical Assessment)',
            'EM 10': 'Đánh giá đồ án (Project Assessment)',
            'EM11': 'Đánh giá thực hành tại phòng thí nghiệm (Practice In The Laboratory Assessment)',
            'EM 12': 'Đánh giá bài tập lớn/Đồ án cá nhân (Major Assignment/Individual Project Assessment)',
            'EM 14': 'Đánh giá khác (Other Assessment)'
        }
        
        # Thêm mapping cho phương pháp giảng dạy (TM)
        self.teaching_methods_mapping = {
            'TM1': 'Giải thích cụ thể (Explicit Teaching)',
            'TM2': 'Thuyết giảng (Lecture)',
            'TM3': 'Tham luận (Guest Lecture)',
            'TM4': 'Câu hỏi gợi mở (Inquiry)',
            'TM5': 'Trò chơi (Game)',
            'TM6': 'Thực hành (Practice)',
            'TM7': 'Thí nghiệm (Experiment)',
            'TM8': 'Thực tập, thực tế (Internship, Field Trip)',
            'TM9': 'Thảo luận (Discussion)',
            'TM10': 'Tranh luận (Debates)',
            'TM11': 'Mô hình (Models)',
            'TM12': 'Mô phỏng (Simulation)',
            'TM13': 'Đóng vai (Role Play)',
            'TM14': 'Giải quyết vấn đề (Problem Solving)',
            'TM15': 'Tập kích não (Brainstorming)',
            'TM16': 'Học theo tình huống (Case Study)',
            'TM17': 'Học nhóm (Teamwork Learning)',
            'TM18': 'Dự án nghiên cứu Đồ án (Project)',
            'TM19': 'Nhóm nghiên cứu giảng dạy (Teaching Research Team)',
            'TM20': 'Học trực tuyến (E-Learning)',
            'TM21': 'Tự học có hướng dẫn (Guided Self-Study)',
            'TM22': 'Bài tập ở nhà (Work Assignment)'
        }
        
        # Mapping giữa PPDG và TM tương thích
        self.ppdg_tm_compatibility = {
            'EM 1': ['TM1', 'TM2', 'TM3', 'TM4'],  # Chuyên cần phù hợp với các phương pháp trực tiếp
            'EM 2': ['TM21', 'TM22'],  # Bài tập cá nhân phù hợp với tự học
            'EM 3': ['TM3', 'TM10', 'TM13'],  # Thuyết trình phù hợp với tham luận, tranh luận, đóng vai
            'EM 4': ['TM9', 'TM17'],  # Làm việc nhóm phù hợp với thảo luận, học nhóm
            'EM 5': ['TM21'],  # Tự học tại thư viện
            'EM 6': ['TM1', 'TM2', 'TM4'],  # Kiểm tra viết phù hợp với giảng dạy trực tiếp
            'EM 7': ['TM1', 'TM2', 'TM4'],  # Trắc nghiệm phù hợp với giảng dạy trực tiếp
            'EM 8': ['TM16', 'TM18', 'TM19'],  # Báo cáo/tiểu luận phù hợp với case study, project
            'EM11': ['TM6', 'TM7', 'TM8'],  # Thực hành phòng thí nghiệm
            'EM 12': ['TM18', 'TM19']  # Đồ án phù hợp với project, research
        }
        
        self.df_ppdg = None
        self.ppdg_model = None
        self.feature_importance = None
        
    def load_ppdg_data(self):
        """Load dữ liệu PPDG"""
        try:
            df_ppdg = pd.read_excel('dulieu/PPDG.xlsx')
            print(f"Đã load dữ liệu PPDG: {len(df_ppdg)} môn học")
            return df_ppdg
        except Exception as e:
            print(f"Lỗi khi load dữ liệu PPDG: {e}")
            return None
    
    def create_ppdg_features(self, df_main, df_ppdg):
        """Tạo features PPDG cho dữ liệu chính"""
        print("=== TẠO FEATURES PPDG ===")
        
        # Merge dữ liệu theo Subject_ID
        if 'Subject_ID' in df_main.columns and 'Subject_ID' in df_ppdg.columns:
            merged_df = df_main.merge(df_ppdg, on='Subject_ID', how='left', suffixes=('', '_ppdg'))
            print(f"Dữ liệu sau khi merge: {len(merged_df)} bản ghi")
            
            # Tìm các cột EM
            em_columns = [col for col in df_ppdg.columns if col.startswith('EM')]
            
            # Tạo features PPDG (1 = có sử dụng, 0 = không sử dụng)
            for col in em_columns:
                merged_df[f'{col}_used'] = merged_df[col].notna().astype(int)
            
            # Tạo features tổng hợp
            ppdg_features = [f'{col}_used' for col in em_columns]
            merged_df['total_ppdg_count'] = merged_df[ppdg_features].sum(axis=1)
            
            # Tạo features theo loại đánh giá
            formative_ppdg = ['EM 1_used', 'EM 2_used', 'EM 3_used', 'EM 4_used', 'EM 5_used']
            summative_ppdg = ['EM 6_used', 'EM 7_used', 'EM 8_used', 'EM 9_used', 'EM 10_used', 'EM11_used', 'EM 12_used', 'EM 14_used']
            
            merged_df['formative_ppdg_count'] = merged_df[formative_ppdg].sum(axis=1)
            merged_df['summative_ppdg_count'] = merged_df[summative_ppdg].sum(axis=1)
            
            print(f"Đã tạo {len(ppdg_features)} features PPDG")
            print(f"Features tổng hợp: total_ppdg_count, formative_ppdg_count, summative_ppdg_count")
            
            return merged_df, ppdg_features
            
        else:
            print("Không tìm thấy cột Subject_ID để merge")
            return df_main, []
    
    def train_ppdg_model(self, df, target_column='exam_score_6'):
        """Train model dự đoán CLO với features PPDG"""
        print("=== TRAIN MODEL PPDG ===")
        
        # Tìm các features PPDG
        ppdg_features = [col for col in df.columns if col.endswith('_used') or 'ppdg_count' in col]
        
        if not ppdg_features:
            print("Không tìm thấy features PPDG")
            return None
        
        # Tạo target binary (1 = đạt CLO, 0 = không đạt)
        if target_column in df.columns:
            df['clo_achieved'] = (df[target_column] >= 3.6).astype(int)
            target = 'clo_achieved'
        else:
            print(f"Không tìm thấy cột {target_column}")
            return None
        
        # Chuẩn bị features
        X = df[ppdg_features].fillna(0)
        y = df[target]
        
        print(f"Features sử dụng: {ppdg_features}")
        print(f"Target: {target}")
        print(f"Số lượng mẫu: {len(X)}")
        print(f"Tỷ lệ đạt CLO: {y.mean():.2%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.ppdg_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ppdg_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ppdg_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': ppdg_features,
            'importance': self.ppdg_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return self.ppdg_model
    
    def explain_clo_with_ppdg(self, student_data, subject_id, predicted_score=None):
        """Giải thích điểm CLO dựa trên PPDG"""
        print(f"\n=== GIẢI THÍCH CLO CHO SINH VIÊN {student_data.get('student_id', 'Unknown')} ===")
        print(f"Môn học: {subject_id}")
        
        if self.ppdg_model is None:
            print("Model PPDG chưa được train")
            return
        
        # Lấy thông tin PPDG của môn học
        ppdg_info = self.get_subject_ppdg_info(subject_id)
        
        if ppdg_info:
            print("\nPhương pháp đánh giá được sử dụng:")
            for ppdg_code, used in ppdg_info.items():
                if used:
                    ppdg_name = self.ppdg_mapping.get(ppdg_code, ppdg_code)
                    print(f"  ✓ {ppdg_code}: {ppdg_name}")
            
            # Phân tích ảnh hưởng cơ bản
            self.analyze_ppdg_impact(ppdg_info, student_data)
            
            # Phân tích hiệu quả và đưa ra khuyến nghị cải thiện
            if predicted_score is not None:
                effectiveness_analysis = self.analyze_ppdg_effectiveness(student_data, subject_id, predicted_score)
                return effectiveness_analysis
        else:
            print("Không tìm thấy thông tin PPDG cho môn học này")
            return None
    
    def get_subject_ppdg_info(self, subject_id):
        """Lấy thông tin PPDG của một môn học"""
        try:
            df_ppdg = pd.read_excel('dulieu/PPDG.xlsx')
            subject_data = df_ppdg[df_ppdg['Subject_ID'] == subject_id]
            
            if len(subject_data) > 0:
                em_columns = [col for col in df_ppdg.columns if col.startswith('EM')]
                ppdg_info = {}
                
                for col in em_columns:
                    value = subject_data[col].iloc[0]
                    # Kiểm tra giá trị 'X', 'x' hoặc không null
                    ppdg_info[col] = (value == 'X' or value == 'x' or (pd.notna(value) and str(value).strip() != '' and str(value).strip() != 'nan'))
                
                return ppdg_info
            else:
                return None
        except Exception as e:
            print(f"Lỗi khi lấy thông tin PPDG: {e}")
            return None
    
    def analyze_ppdg_impact(self, ppdg_info, student_data):
        """Phân tích ảnh hưởng của PPDG đến CLO"""
        print("\n=== PHÂN TÍCH ẢNH HƯỞNG PPDG ===")
        
        if self.feature_importance is None:
            print("Chưa có thông tin feature importance")
            return
        
        # Phân tích theo loại PPDG
        formative_impact = []
        summative_impact = []
        
        for ppdg_code, used in ppdg_info.items():
            if used:
                feature_name = f'{ppdg_code}_used'
                importance = self.feature_importance[
                    self.feature_importance['feature'] == feature_name
                ]['importance'].values
                
                if len(importance) > 0:
                    impact = importance[0]
                    ppdg_name = self.ppdg_mapping.get(ppdg_code, ppdg_code)
                    
                    if ppdg_code in ['EM 1', 'EM 2', 'EM 3', 'EM 4', 'EM 5']:
                        formative_impact.append((ppdg_code, ppdg_name, impact))
                    else:
                        summative_impact.append((ppdg_code, ppdg_name, impact))
        
        # In kết quả phân tích
        if formative_impact:
            print("\n1. Đánh giá quá trình (Formative Assessment):")
            for code, name, impact in sorted(formative_impact, key=lambda x: x[2], reverse=True):
                print(f"   - {code} ({name}): Ảnh hưởng {impact:.3f}")
        
        if summative_impact:
            print("\n2. Đánh giá tổng kết (Summative Assessment):")
            for code, name, impact in sorted(summative_impact, key=lambda x: x[2], reverse=True):
                print(f"   - {code} ({name}): Ảnh hưởng {impact:.3f}")
        
        # Đưa ra khuyến nghị
        self.generate_recommendations(ppdg_info, student_data)
    
    def generate_recommendations(self, ppdg_info, student_data):
        """Tạo khuyến nghị dựa trên PPDG"""
        print("\n=== KHUYẾN NGHỊ ===")
        
        # Đếm số lượng PPDG
        total_ppdg = sum(ppdg_info.values())
        formative_count = sum(1 for code, used in ppdg_info.items() 
                            if used and code in ['EM 1', 'EM 2', 'EM 3', 'EM 4', 'EM 5'])
        summative_count = total_ppdg - formative_count
        
        print(f"Số lượng PPDG: {total_ppdg}")
        print(f"  - Đánh giá quá trình: {formative_count}")
        print(f"  - Đánh giá tổng kết: {summative_count}")
        
        # Khuyến nghị dựa trên loại PPDG
        if formative_count >= 3:
            print("\n✓ Môn học này có nhiều đánh giá quá trình → Tập trung vào:")
            print("  - Tham gia đầy đủ các buổi học")
            print("  - Hoàn thành bài tập cá nhân đúng hạn")
            print("  - Tham gia thuyết trình và làm việc nhóm tích cực")
        
        if summative_count >= 2:
            print("\n✓ Môn học này có nhiều đánh giá tổng kết → Tập trung vào:")
            print("  - Ôn tập kỹ lưỡng cho các bài kiểm tra")
            print("  - Chuẩn bị tốt cho báo cáo/tiểu luận")
            print("  - Thực hành kỹ năng thực tế")
        
        # Khuyến nghị cụ thể cho từng PPDG
        specific_recommendations = {
            'EM 1': "Đi học đầy đủ và đúng giờ",
            'EM 2': "Hoàn thành bài tập cá nhân chất lượng cao",
            'EM 3': "Chuẩn bị kỹ lưỡng cho thuyết trình",
            'EM 4': "Tích cực tham gia làm việc nhóm",
            'EM 5': "Tăng cường tự học tại thư viện",
            'EM 6': "Ôn tập kỹ cho bài kiểm tra viết",
            'EM 7': "Luyện tập kỹ năng làm bài trắc nghiệm",
            'EM 8': "Viết báo cáo/tiểu luận chất lượng cao",
            'EM11': "Thực hành kỹ lưỡng tại phòng thí nghiệm",
            'EM 12': "Hoàn thành bài tập lớn/đồ án đúng hạn"
        }
        
        print("\nKhuyến nghị cụ thể:")
        for ppdg_code, used in ppdg_info.items():
            if used and ppdg_code in specific_recommendations:
                print(f"  - {ppdg_code}: {specific_recommendations[ppdg_code]}")

    def analyze_ppdg_effectiveness(self, student_data, subject_id, predicted_score, teaching_methods=None):
        """Phân tích hiệu quả của PPDG và đưa ra khuyến nghị cải thiện"""
        print("\n=== PHÂN TÍCH HIỆU QUẢ PPDG ===")
        
        # Lấy thông tin PPDG của môn học
        ppdg_info = self.get_subject_ppdg_info(subject_id)
        
        if not ppdg_info:
            print("Không tìm thấy thông tin PPDG cho môn học này")
            return
        
        # Phân tích hiện trạng PPDG
        current_ppdg_analysis = self.analyze_current_ppdg_status(ppdg_info)
        
        # Đánh giá hiệu quả dựa trên điểm dự đoán
        effectiveness_analysis = self.evaluate_ppdg_effectiveness(ppdg_info, predicted_score)
        
        # Phân tích tương thích với phương pháp giảng dạy
        compatibility_analysis = self.analyze_teaching_method_compatibility(ppdg_info, teaching_methods)
        
        # Đề xuất cải thiện phương pháp giảng dạy
        teaching_improvements = self.suggest_teaching_method_improvements(ppdg_info, compatibility_analysis)
        
        # Đưa ra khuyến nghị cải thiện PPDG
        improvement_recommendations = self.generate_improvement_recommendations(
            ppdg_info, current_ppdg_analysis, effectiveness_analysis, predicted_score
        )
        
        return {
            'current_status': current_ppdg_analysis,
            'effectiveness': effectiveness_analysis,
            'compatibility': compatibility_analysis,
            'teaching_improvements': teaching_improvements,
            'recommendations': improvement_recommendations
        }
    
    def analyze_teaching_method_compatibility(self, ppdg_info, teaching_methods=None):
        """Phân tích tính tương thích giữa PPDG và phương pháp giảng dạy"""
        print(f"\n🎯 PHÂN TÍCH TƯƠNG THÍCH PPDG - PHƯƠNG PHÁP GIẢNG DẠY:")
        
        if not teaching_methods:
            # Giả định các phương pháp giảng dạy phổ biến nếu không có dữ liệu
            teaching_methods = ['TM1', 'TM2', 'TM4', 'TM9', 'TM21']
            print("⚠️ Sử dụng phương pháp giảng dạy mặc định (cần dữ liệu thực tế)")
        
        compatibility_score = 0
        total_ppdg = len([code for code, used in ppdg_info.items() if used])
        compatible_ppdg = 0
        incompatible_ppdg = []
        
        print(f"\n📋 PHƯƠNG PHÁP GIẢNG DẠY ĐANG SỬ DỤNG:")
        for tm in teaching_methods:
            tm_name = self.teaching_methods_mapping.get(tm, tm)
            print(f"  • {tm}: {tm_name}")
        
        print(f"\n🔍 KIỂM TRA TƯƠNG THÍCH:")
        for ppdg_code, used in ppdg_info.items():
            if used:
                compatible_tm = self.ppdg_tm_compatibility.get(ppdg_code, [])
                ppdg_name = self.ppdg_mapping.get(ppdg_code, ppdg_code)
                
                # Kiểm tra xem có TM tương thích không
                has_compatible_tm = any(tm in teaching_methods for tm in compatible_tm)
                
                if has_compatible_tm:
                    compatible_ppdg += 1
                    compatible_tm_names = [self.teaching_methods_mapping.get(tm, tm) for tm in compatible_tm if tm in teaching_methods]
                    print(f"  ✅ {ppdg_code}: {ppdg_name}")
                    print(f"     Tương thích với: {', '.join(compatible_tm_names)}")
                else:
                    incompatible_ppdg.append(ppdg_code)
                    print(f"  ❌ {ppdg_code}: {ppdg_name}")
                    print(f"     Không tương thích với phương pháp giảng dạy hiện tại")
        
        # Tính điểm tương thích
        if total_ppdg > 0:
            compatibility_score = (compatible_ppdg / total_ppdg) * 10
        
        print(f"\n📊 KẾT QUẢ TƯƠNG THÍCH:")
        print(f"• Số PPDG tương thích: {compatible_ppdg}/{total_ppdg}")
        print(f"• Điểm tương thích: {compatibility_score:.1f}/10")
        
        if incompatible_ppdg:
            print(f"• PPDG không tương thích: {', '.join(incompatible_ppdg)}")
        
        # Đánh giá mức độ tương thích
        if compatibility_score >= 8:
            compatibility_level = "CAO"
            compatibility_desc = "PPDG và phương pháp giảng dạy rất tương thích"
        elif compatibility_score >= 6:
            compatibility_level = "TRUNG BÌNH"
            compatibility_desc = "PPDG và phương pháp giảng dạy tương thích vừa phải"
        else:
            compatibility_level = "THẤP"
            compatibility_desc = "PPDG và phương pháp giảng dạy ít tương thích"
        
        print(f"• Mức độ tương thích: {compatibility_level}")
        print(f"• Đánh giá: {compatibility_desc}")
        
        return {
            'compatibility_score': compatibility_score,
            'compatibility_level': compatibility_level,
            'compatible_ppdg_count': compatible_ppdg,
            'total_ppdg_count': total_ppdg,
            'incompatible_ppdg': incompatible_ppdg,
            'teaching_methods': teaching_methods
        }
    
    def suggest_teaching_method_improvements(self, ppdg_info, compatibility_analysis):
        """Đề xuất cải thiện phương pháp giảng dạy dựa trên PPDG"""
        print(f"\n🔧 ĐỀ XUẤT CẢI THIỆN PHƯƠNG PHÁP GIẢNG DẠY:")
        
        suggestions = []
        
        # Kiểm tra PPDG không tương thích
        for ppdg_code in compatibility_analysis['incompatible_ppdg']:
            compatible_tm = self.ppdg_tm_compatibility.get(ppdg_code, [])
            ppdg_name = self.ppdg_mapping.get(ppdg_code, ppdg_code)
            
            if compatible_tm:
                suggested_tm = []
                for tm in compatible_tm:
                    tm_name = self.teaching_methods_mapping.get(tm, tm)
                    suggested_tm.append(f"{tm} ({tm_name})")
                
                suggestions.append({
                    'type': 'ADD_TEACHING_METHOD',
                    'ppdg': ppdg_code,
                    'ppdg_name': ppdg_name,
                    'suggested_tm': suggested_tm,
                    'priority': 'CAO' if ppdg_code in ['EM 1', 'EM 2', 'EM 3'] else 'TRUNG BÌNH'
                })
        
        # Đề xuất PPDG bổ sung cho phương pháp giảng dạy hiện tại
        current_tm = compatibility_analysis['teaching_methods']
        missing_ppdg = []
        
        for tm in current_tm:
            for ppdg_code, compatible_tm_list in self.ppdg_tm_compatibility.items():
                if tm in compatible_tm_list and not ppdg_info.get(ppdg_code, False):
                    ppdg_name = self.ppdg_mapping.get(ppdg_code, ppdg_code)
                    tm_name = self.teaching_methods_mapping.get(tm, tm)
                    missing_ppdg.append({
                        'ppdg_code': ppdg_code,
                        'ppdg_name': ppdg_name,
                        'tm_code': tm,
                        'tm_name': tm_name
                    })
        
        if missing_ppdg:
            # Nhóm theo PPDG
            ppdg_groups = {}
            for item in missing_ppdg:
                ppdg_code = item['ppdg_code']
                if ppdg_code not in ppdg_groups:
                    ppdg_groups[ppdg_code] = []
                ppdg_groups[ppdg_code].append(item)
            
            for ppdg_code, items in ppdg_groups.items():
                ppdg_name = items[0]['ppdg_name']
                tm_suggestions = [f"{item['tm_code']} ({item['tm_name']})" for item in items]
                
                suggestions.append({
                    'type': 'ADD_PPDG',
                    'ppdg': ppdg_code,
                    'ppdg_name': ppdg_name,
                    'suggested_tm': tm_suggestions,
                    'priority': 'TRUNG BÌNH'
                })
        
        # In đề xuất
        if suggestions:
            print(f"\n📋 CÁC ĐỀ XUẤT CẢI THIỆN:")
            for i, suggestion in enumerate(suggestions, 1):
                priority_icon = "🔴" if suggestion['priority'] == 'CAO' else "🟡" if suggestion['priority'] == 'TRUNG BÌNH' else "🟢"
                
                if suggestion['type'] == 'ADD_TEACHING_METHOD':
                    print(f"{i}. {priority_icon} Bổ sung phương pháp giảng dạy cho {suggestion['ppdg']} ({suggestion['ppdg_name']})")
                    print(f"   Đề xuất: {', '.join(suggestion['suggested_tm'])}")
                
                elif suggestion['type'] == 'ADD_PPDG':
                    print(f"{i}. {priority_icon} Bổ sung PPDG {suggestion['ppdg']} ({suggestion['ppdg_name']})")
                    print(f"   Phù hợp với: {', '.join(suggestion['suggested_tm'])}")
                
                print(f"   Ưu tiên: {suggestion['priority']}")
                print()
        else:
            print("✅ Không có đề xuất cải thiện - PPDG và phương pháp giảng dạy đã tương thích tốt!")
        
        return suggestions
    
    def analyze_current_ppdg_status(self, ppdg_info):
        """Phân tích hiện trạng PPDG của môn học"""
        print("\n📊 HIỆN TRẠNG PPDG:")
        
        # Đếm số lượng PPDG
        total_ppdg = sum(ppdg_info.values())
        formative_count = sum(1 for code, used in ppdg_info.items() 
                            if used and code in ['EM 1', 'EM 2', 'EM 3', 'EM 4', 'EM 5'])
        summative_count = total_ppdg - formative_count
        
        print(f"• Tổng số PPDG: {total_ppdg}")
        print(f"• Đánh giá quá trình: {formative_count}")
        print(f"• Đánh giá tổng kết: {summative_count}")
        
        # Phân tích chi tiết từng PPDG
        print("\n📋 CHI TIẾT CÁC PPDG:")
        for ppdg_code, used in ppdg_info.items():
            if used:
                ppdg_name = self.ppdg_mapping.get(ppdg_code, ppdg_code)
                print(f"  ✓ {ppdg_code}: {ppdg_name}")
        
        # Đánh giá mức độ đa dạng
        diversity_score = self.calculate_ppdg_diversity(ppdg_info)
        print(f"\n🎯 ĐIỂM ĐA DẠNG PPDG: {diversity_score:.1f}/10")
        
        return {
            'total_ppdg': total_ppdg,
            'formative_count': formative_count,
            'summative_count': summative_count,
            'diversity_score': diversity_score,
            'used_ppdg': [code for code, used in ppdg_info.items() if used]
        }
    
    def calculate_ppdg_diversity(self, ppdg_info):
        """Tính điểm đa dạng của PPDG (0-10)"""
        used_ppdg = [code for code, used in ppdg_info.items() if used]
        
        if not used_ppdg:
            return 0
        
        # Điểm cơ bản dựa trên số lượng PPDG
        base_score = min(len(used_ppdg) * 1.5, 6)  # Tối đa 6 điểm cho số lượng
        
        # Điểm bổ sung cho sự cân bằng
        formative_count = sum(1 for code in used_ppdg if code in ['EM 1', 'EM 2', 'EM 3', 'EM 4', 'EM 5'])
        summative_count = len(used_ppdg) - formative_count
        
        balance_score = 0
        if formative_count > 0 and summative_count > 0:
            balance_score = 2  # Có cả đánh giá quá trình và tổng kết
        elif formative_count >= 3:
            balance_score = 1.5  # Nhiều đánh giá quá trình
        elif summative_count >= 2:
            balance_score = 1  # Nhiều đánh giá tổng kết
        
        # Điểm cho các PPDG đặc biệt
        special_score = 0
        special_ppdg = ['EM 3', 'EM 4', 'EM 8', 'EM11', 'EM 12']  # Các PPDG phát triển kỹ năng
        for code in used_ppdg:
            if code in special_ppdg:
                special_score += 0.5
        
        total_score = min(base_score + balance_score + special_score, 10)
        return total_score
    
    def evaluate_ppdg_effectiveness(self, ppdg_info, predicted_score):
        """Đánh giá hiệu quả của PPDG dựa trên điểm dự đoán"""
        print(f"\n📈 ĐÁNH GIÁ HIỆU QUẢ PPDG (Điểm dự đoán: {predicted_score:.2f}/6):")
        
        # Phân tích dựa trên điểm dự đoán
        if predicted_score >= 4.5:
            effectiveness_level = "CAO"
            effectiveness_desc = "PPDG hiện tại có hiệu quả tốt"
        elif predicted_score >= 3.6:
            effectiveness_level = "TRUNG BÌNH"
            effectiveness_desc = "PPDG có hiệu quả vừa phải, có thể cải thiện"
        else:
            effectiveness_level = "THẤP"
            effectiveness_desc = "PPDG cần được cải thiện đáng kể"
        
        print(f"• Mức độ hiệu quả: {effectiveness_level}")
        print(f"• Đánh giá: {effectiveness_desc}")
        
        # Phân tích điểm mạnh và điểm yếu
        strengths, weaknesses = self.identify_ppdg_strengths_weaknesses(ppdg_info, predicted_score)
        
        if strengths:
            print("\n✅ ĐIỂM MẠNH:")
            for strength in strengths:
                print(f"  - {strength}")
        
        if weaknesses:
            print("\n❌ ĐIỂM YẾU:")
            for weakness in weaknesses:
                print(f"  - {weakness}")
        
        return {
            'level': effectiveness_level,
            'description': effectiveness_desc,
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def identify_ppdg_strengths_weaknesses(self, ppdg_info, predicted_score):
        """Xác định điểm mạnh và điểm yếu của PPDG"""
        strengths = []
        weaknesses = []
        
        used_ppdg = [code for code, used in ppdg_info.items() if used]
        total_ppdg = len(used_ppdg)
        
        # Phân tích số lượng PPDG
        if total_ppdg >= 6:
            strengths.append("Sử dụng nhiều PPDG đa dạng")
        elif total_ppdg <= 3:
            weaknesses.append("Sử dụng ít PPDG, thiếu đa dạng")
        
        # Phân tích cân bằng đánh giá
        formative_count = sum(1 for code in used_ppdg if code in ['EM 1', 'EM 2', 'EM 3', 'EM 4', 'EM 5'])
        summative_count = total_ppdg - formative_count
        
        if formative_count >= 3 and summative_count >= 2:
            strengths.append("Cân bằng tốt giữa đánh giá quá trình và tổng kết")
        elif formative_count < 2:
            weaknesses.append("Thiếu đánh giá quá trình, sinh viên ít được theo dõi liên tục")
        elif summative_count < 1:
            weaknesses.append("Thiếu đánh giá tổng kết, khó đánh giá toàn diện")
        
        # Phân tích các PPDG đặc biệt
        special_ppdg = ['EM 3', 'EM 4', 'EM 8', 'EM11', 'EM 12']
        special_count = sum(1 for code in used_ppdg if code in special_ppdg)
        
        if special_count >= 2:
            strengths.append("Có nhiều PPDG phát triển kỹ năng thực tế")
        elif special_count == 0:
            weaknesses.append("Thiếu PPDG phát triển kỹ năng thực tế")
        
        # Phân tích dựa trên điểm dự đoán
        if predicted_score < 3.6:
            if 'EM 1' not in used_ppdg:
                weaknesses.append("Thiếu đánh giá chuyên cần (EM 1) - yếu tố quan trọng cho sinh viên yếu")
            if 'EM 2' not in used_ppdg:
                weaknesses.append("Thiếu đánh giá bài tập cá nhân (EM 2) - giúp sinh viên luyện tập")
        
        return strengths, weaknesses
    
    def generate_improvement_recommendations(self, ppdg_info, current_status, effectiveness, predicted_score):
        """Tạo khuyến nghị cải thiện PPDG"""
        print(f"\n🔧 KHUYẾN NGHỊ CẢI THIỆN PPDG:")
        
        recommendations = []
        priority_levels = []
        
        # Khuyến nghị dựa trên số lượng PPDG
        if current_status['total_ppdg'] < 5:
            recommendations.append("Tăng số lượng PPDG lên ít nhất 5-6 phương pháp")
            priority_levels.append("CAO")
        
        # Khuyến nghị dựa trên cân bằng đánh giá
        if current_status['formative_count'] < 2:
            recommendations.append("Bổ sung thêm đánh giá quá trình (EM 1, EM 2, EM 3, EM 4, EM 5)")
            priority_levels.append("CAO")
        
        if current_status['summative_count'] < 2:
            recommendations.append("Bổ sung thêm đánh giá tổng kết (EM 6, EM 7, EM 8, EM11, EM 12)")
            priority_levels.append("TRUNG BÌNH")
        
        # Khuyến nghị dựa trên điểm dự đoán
        if predicted_score < 3.6:
            if 'EM 1' not in ppdg_info or not ppdg_info['EM 1']:
                recommendations.append("Thêm đánh giá chuyên cần (EM 1) để theo dõi sự tham gia")
                priority_levels.append("CAO")
            
            if 'EM 2' not in ppdg_info or not ppdg_info['EM 2']:
                recommendations.append("Thêm đánh giá bài tập cá nhân (EM 2) để luyện tập thường xuyên")
                priority_levels.append("CAO")
            
            if 'EM 3' not in ppdg_info or not ppdg_info['EM 3']:
                recommendations.append("Thêm đánh giá thuyết trình (EM 3) để phát triển kỹ năng giao tiếp")
                priority_levels.append("TRUNG BÌNH")
        
        # Khuyến nghị dựa trên đa dạng
        if current_status['diversity_score'] < 6:
            recommendations.append("Tăng tính đa dạng của PPDG để phát triển toàn diện")
            priority_levels.append("TRUNG BÌNH")
        
        # Khuyến nghị cụ thể cho từng PPDG thiếu
        missing_ppdg = []
        for code, used in ppdg_info.items():
            if not used and code in ['EM 1', 'EM 2', 'EM 3', 'EM 4', 'EM 5', 'EM 8', 'EM11']:
                ppdg_name = self.ppdg_mapping.get(code, code)
                missing_ppdg.append(f"{code} ({ppdg_name})")
        
        if missing_ppdg:
            recommendations.append(f"Cân nhắc bổ sung: {', '.join(missing_ppdg[:3])}")
            priority_levels.append("THẤP")
        
        # In khuyến nghị
        for i, (rec, priority) in enumerate(zip(recommendations, priority_levels), 1):
            priority_icon = "🔴" if priority == "CAO" else "🟡" if priority == "TRUNG BÌNH" else "🟢"
            print(f"{i}. {priority_icon} {rec} (Ưu tiên: {priority})")
        
        # Tóm tắt
        print(f"\n📋 TÓM TẮT:")
        print(f"• Hiện tại: {current_status['total_ppdg']} PPDG (Điểm đa dạng: {current_status['diversity_score']:.1f}/10)")
        print(f"• Hiệu quả: {effectiveness['level']}")
        print(f"• Số khuyến nghị: {len(recommendations)}")
        
        return {
            'recommendations': recommendations,
            'priorities': priority_levels,
            'summary': {
                'current_ppdg_count': current_status['total_ppdg'],
                'diversity_score': current_status['diversity_score'],
                'effectiveness_level': effectiveness['level'],
                'recommendation_count': len(recommendations)
            }
        }

def integrate_ppdg_with_main_system():
    """Tích hợp PPDG vào hệ thống chính"""
    print("=== TÍCH HỢP PPDG VÀO HỆ THỐNG CHÍNH ===")
    
    try:
        # Import các module chính
        from data_loader import DataLoader
        from model_trainer import ModelTrainer
        
        # Load dữ liệu
        data_loader = DataLoader()
        df_main = data_loader.df
        
        # Tạo PPDG integration
        ppdg_integration = PPDGIntegration()
        df_ppdg = ppdg_integration.load_ppdg_data()
        
        if df_ppdg is not None:
            # Tạo features PPDG
            df_with_ppdg, ppdg_features = ppdg_integration.create_ppdg_features(df_main, df_ppdg)
            
            # Train model PPDG
            ppdg_model = ppdg_integration.train_ppdg_model(df_with_ppdg)
            
            if ppdg_model:
                print("\n=== TÍCH HỢP THÀNH CÔNG ===")
                print("PPDG đã được tích hợp vào hệ thống dự đoán CLO")
                print("Có thể sử dụng ppdg_integration.explain_clo_with_ppdg() để giải thích điểm CLO")
                
                return ppdg_integration
            else:
                print("Không thể train model PPDG")
                return None
        else:
            print("Không thể load dữ liệu PPDG")
            return None
            
    except ImportError as e:
        print(f"Lỗi import: {e}")
        return None

if __name__ == "__main__":
    # Test tích hợp PPDG
    ppdg_integration = integrate_ppdg_with_main_system()
    
    if ppdg_integration:
        # Test giải thích CLO
        test_student = {'student_id': 'TEST001'}
        test_subject = 'PLO0043'  # Triết học Mác - Lênin
        ppdg_integration.explain_clo_with_ppdg(test_student, test_subject) 