import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.df = data_loader.df

    def add_student_history_features(self):
        """Tính toán lịch sử học tập cho từng sinh viên"""
        print("Adding student history features...")
        
        # Calculate student history features
        student_history = self.df.groupby('Student_ID').agg({
            'passed': ['count', 'sum', 'mean'],
            'clo_achieved': ['sum', 'mean'],
            'exam_score_6': ['mean', 'std', 'min', 'max'],
            'summary_score_numeric': ['mean', 'std'],
            'is_absent_summary': 'sum',
            'is_absent_exam': 'sum'
        }).reset_index()
        
        student_history.columns = [
            'Student_ID', 'total_subjects', 'passed_subjects', 'pass_rate',
            'clo_achieved_count', 'clo_achieved_rate', 'avg_exam_score',
            'std_exam_score', 'min_exam_score', 'max_exam_score',
            'avg_summary_score', 'std_summary_score', 'absent_summary_count',
            'absent_exam_count'
        ]
        
        # Merge back to main dataframe
        self.df = pd.merge(self.df, student_history, on='Student_ID', how='left')
        
        # Add these features to feature list
        history_features = [
            'total_subjects', 'passed_subjects', 'pass_rate',
            'clo_achieved_count', 'clo_achieved_rate', 'avg_exam_score',
            'std_exam_score', 'min_exam_score', 'max_exam_score',
            'avg_summary_score', 'std_summary_score', 'absent_summary_count',
            'absent_exam_count'
        ]
        
        # Only add features that exist in the dataframe
        available_history_features = [col for col in history_features if col in self.df.columns]
        self.data_loader.feature_names.extend(available_history_features)

    def add_advanced_student_features(self):
        """Add advanced student features"""
        print("Adding advanced student features...")
        
        # Calculate recent performance (last 3 subjects)
        def recent_performance(student_data):
            if len(student_data) <= 3:
                return student_data['exam_score_6'].mean()
            return student_data.sort_values('year').tail(3)['exam_score_6'].mean()
        
        recent_scores = self.df.groupby('Student_ID').apply(recent_performance).reset_index()
        recent_scores.columns = ['Student_ID', 'recent_avg_score']
        
        # Calculate improvement trend
        def improvement_trend(student_data):
            if len(student_data) < 2:
                return 0
            sorted_data = student_data.sort_values('year')
            if len(sorted_data) >= 2:
                recent = sorted_data.tail(2)['exam_score_6'].mean()
                earlier = sorted_data.head(len(sorted_data)-2)['exam_score_6'].mean()
                return recent - earlier
            return 0
        
        improvement_trends = self.df.groupby('Student_ID').apply(improvement_trend).reset_index()
        improvement_trends.columns = ['Student_ID', 'improvement_trend']
        
        # Merge features
        self.df = pd.merge(self.df, recent_scores, on='Student_ID', how='left')
        self.df = pd.merge(self.df, improvement_trends, on='Student_ID', how='left')
        
        # Add to feature list
        advanced_features = ['recent_avg_score', 'improvement_trend']
        available_advanced_features = [col for col in advanced_features if col in self.df.columns]
        self.data_loader.feature_names.extend(available_advanced_features)

    def add_personalized_features(self):
        """Add personalized features based on student characteristics"""
        print("Adding personalized features...")
        
        def recent_pass_count(row):
            student_data = self.df[self.df['Student_ID'] == row['Student_ID']]
            recent_data = student_data.sort_values('year').tail(3)
            return recent_data['passed'].sum()
        
        def recent_fail_count(row):
            student_data = self.df[self.df['Student_ID'] == row['Student_ID']]
            recent_data = student_data.sort_values('year').tail(3)
            return (recent_data['passed'] == 0).sum()
        
        def recent_avg_score(row):
            student_data = self.df[self.df['Student_ID'] == row['Student_ID']]
            recent_data = student_data.sort_values('year').tail(3)
            return recent_data['exam_score_6'].mean()
        
        def num_with_lecturer(row):
            return len(self.df[(self.df['Student_ID'] == row['Student_ID']) & 
                             (self.df['Lecturer_Name'] == row['Lecturer_Name'])])
        
        def num_in_group(row):
            return len(self.df[(self.df['Student_ID'] == row['Student_ID']) & 
                             (self.df['Subject_ID'] == row['Subject_ID'])])
        
        # Apply functions
        self.df['recent_pass_count'] = self.df.apply(recent_pass_count, axis=1)
        self.df['recent_fail_count'] = self.df.apply(recent_fail_count, axis=1)
        self.df['recent_avg_score'] = self.df.apply(recent_avg_score, axis=1)
        self.df['num_with_lecturer'] = self.df.apply(num_with_lecturer, axis=1)
        self.df['num_in_group'] = self.df.apply(num_in_group, axis=1)
        
        # Add to feature list
        personalized_features = ['recent_pass_count', 'recent_fail_count', 'num_with_lecturer', 'num_in_group']
        available_personalized_features = [col for col in personalized_features if col in self.df.columns]
        self.data_loader.feature_names.extend(available_personalized_features)

    def print_demographic_statistics(self):
        """Print demographic statistics"""
        print("\n===== THỐNG KÊ NHÂN KHẨU HỌC TOÀN BỘ DỮ LIỆU =====")
        demo_features = [
            ('gender_encoded', self.data_loader.gender_col, 'Giới tính'),
            ('religion_encoded', self.data_loader.religion_col, 'Tôn giáo'),
            ('birth_place_encoded', self.data_loader.birth_place_col, 'Nơi sinh'),
            ('ethnicity_encoded', self.data_loader.ethnicity_col, 'Dân tộc')
        ]
        
        for feat, col, label in demo_features:
            if col and feat in self.df.columns:
                print(f"\n--- {label} ---")
                group_stats = self.df.groupby(feat).agg(
                    so_luong = ('Student_ID', 'count'),
                    diem_tb = ('exam_score_6', 'mean'),
                    ti_le_pass = ('passed', 'mean')
                ).reset_index()
                
                # Get group names from demographic file if available
                if hasattr(self.data_loader, 'le_' + feat.split('_')[0]):
                    le = getattr(self.data_loader, 'le_' + feat.split('_')[0])
                    try:
                        group_stats[label] = le.inverse_transform(group_stats[feat])
                    except:
                        group_stats[label] = group_stats[feat]
                else:
                    group_stats[label] = group_stats[feat]
                
                for _, row in group_stats.iterrows():
                    print(f"{label}: {row[label]} | Số lượng: {int(row['so_luong'])} | Điểm TB: {row['diem_tb']:.2f}/6 | Tỉ lệ pass: {row['ti_le_pass']*100:.1f}%") 