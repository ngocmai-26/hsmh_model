import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataIntegration:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.df = data_loader.df

    def integrate_demographic_data(self):
        """Tích hợp dữ liệu nhân khẩu vào dữ liệu chính"""
        print("Integrating demographic data...")
        
        if self.data_loader.nhankhau_df is None:
            print("No demographic data available")
            return
        
        # Initialize demographic columns with default values
        self.df['gender_encoded'] = -1
        self.df['religion_encoded'] = -1
        self.df['birth_place_encoded'] = -1
        self.df['ethnicity_encoded'] = -1
        
        # Create mapping from Student_ID to demographic information
        demographic_mapping = {}
        for _, row in self.data_loader.nhankhau_df.iterrows():
            student_id = row['Student_ID']
            demographic_info = {}
            
            # Process gender
            if self.data_loader.gender_col and pd.notna(row[self.data_loader.gender_col]):
                gender_value = str(row[self.data_loader.gender_col]).strip()
                if gender_value in ['Nam', 'Male', '1']:
                    demographic_info['gender'] = 'Nam'
                elif gender_value in ['Nữ', 'Female', '0']:
                    demographic_info['gender'] = 'Nữ'
                else:
                    demographic_info['gender'] = 'Khác'
            
            # Process religion
            if self.data_loader.religion_col and pd.notna(row[self.data_loader.religion_col]):
                religion_value = str(row[self.data_loader.religion_col]).strip()
                if religion_value in ['Không', 'None', '']:
                    demographic_info['religion'] = 'Không tôn giáo'
                else:
                    demographic_info['religion'] = religion_value
            
            # Process birth place
            if self.data_loader.birth_place_col and pd.notna(row[self.data_loader.birth_place_col]):
                birth_place_value = str(row[self.data_loader.birth_place_col]).strip()
                demographic_info['birth_place'] = birth_place_value
            
            # Process ethnicity
            if self.data_loader.ethnicity_col and pd.notna(row[self.data_loader.ethnicity_col]):
                ethnicity_value = str(row[self.data_loader.ethnicity_col]).strip()
                if ethnicity_value in ['Kinh', 'Việt']:
                    demographic_info['ethnicity'] = 'Kinh'
                else:
                    demographic_info['ethnicity'] = ethnicity_value
            
            if demographic_info:
                demographic_mapping[student_id] = demographic_info
        
        # Integrate demographic information into main dataframe
        matched_count = 0
        for idx, row in self.df.iterrows():
            student_id = row['Student_ID']
            if student_id in demographic_mapping:
                demo_info = demographic_mapping[student_id]
                matched_count += 1
                
                # Assign demographic values
                if 'gender' in demo_info:
                    self.df.at[idx, 'gender_encoded'] = demo_info['gender']
                if 'religion' in demo_info:
                    self.df.at[idx, 'religion_encoded'] = demo_info['religion']
                if 'birth_place' in demo_info:
                    self.df.at[idx, 'birth_place_encoded'] = demo_info['birth_place']
                if 'ethnicity' in demo_info:
                    self.df.at[idx, 'ethnicity_encoded'] = demo_info['ethnicity']
        
        # Encode demographic variables
        if self.df['gender_encoded'].nunique() > 1:
            self.df['gender_encoded'] = self.data_loader.le_gender.fit_transform(self.df['gender_encoded'].astype(str))
        else:
            self.df['gender_encoded'] = 0
        
        if self.df['religion_encoded'].nunique() > 1:
            self.df['religion_encoded'] = self.data_loader.le_religion.fit_transform(self.df['religion_encoded'].astype(str))
        else:
            self.df['religion_encoded'] = 0
        
        if self.df['birth_place_encoded'].nunique() > 1:
            self.df['birth_place_encoded'] = self.data_loader.le_birth_place.fit_transform(self.df['birth_place_encoded'].astype(str))
        else:
            self.df['birth_place_encoded'] = 0
        
        if self.df['ethnicity_encoded'].nunique() > 1:
            self.df['ethnicity_encoded'] = self.data_loader.le_ethnicity.fit_transform(self.df['ethnicity_encoded'].astype(str))
        else:
            self.df['ethnicity_encoded'] = 0
        
        # Create demographic feature list
        self.data_loader.demographic_features = ['gender_encoded', 'religion_encoded', 'birth_place_encoded', 'ethnicity_encoded']
        
        print(f"Successfully integrated demographic data for {matched_count} student records")
        print(f"Demographic features added: {self.data_loader.demographic_features}")

    def integrate_conduct_data(self):
        """Tích hợp dữ liệu điểm rèn luyện vào dữ liệu chính (theo từng học kỳ, năm học)"""
        print("Integrating conduct score data...")
        
        if self.data_loader.conduct_df is None:
            print("No conduct data available")
            return
        
        try:
            # Normalize data types
            self.data_loader.conduct_df['Student_ID'] = self.data_loader.conduct_df['Student_ID'].astype(str)
            self.data_loader.conduct_df['school_year'] = self.data_loader.conduct_df['school_year'].astype(str)
            self.data_loader.conduct_df['semester'] = self.data_loader.conduct_df['semester'].astype(str)
            self.df['Student_ID'] = self.df['Student_ID'].astype(str)
            self.df['school_year'] = self.df['year'].astype(str) if 'year' in self.df.columns else self.df['school_year'].astype(str)
            self.df['semester'] = self.df['semester'].astype(str) if 'semester' in self.df.columns else '1'

            def get_conduct_row(sid, year, semester):
                df = self.data_loader.conduct_df[self.data_loader.conduct_df['Student_ID'] == sid]
                if df.empty:
                    return None
                df = df[(df['school_year'] < year) | ((df['school_year'] == year) & (df['semester'] <= semester))]
                if df.empty:
                    return None
                return df.sort_values(['school_year', 'semester'], ascending=[False, False]).iloc[0]

            avg_conduct_scores = []
            latest_conduct_scores = []
            latest_conduct_semesters = []
            latest_conduct_years = []
            conduct_trends = []
            latest_classifications = []
            num_conduct_semesters = []

            for idx, row in self.df.iterrows():
                sid = row['Student_ID']
                year = row['school_year']
                semester = row['semester']
                student_conduct = self.data_loader.conduct_df[self.data_loader.conduct_df['Student_ID'] == sid]
                avg_conduct = student_conduct['conduct_score'].mean() if not student_conduct.empty else 65.0
                avg_conduct_scores.append(avg_conduct)
                conduct_row = get_conduct_row(sid, year, semester)
                if conduct_row is not None:
                    latest_conduct_scores.append(conduct_row['conduct_score'])
                    latest_conduct_semesters.append(conduct_row['semester'])
                    latest_conduct_years.append(conduct_row['school_year'])
                    latest_classifications.append(conduct_row['student_conduct_classification'])
                    prev = student_conduct[(student_conduct['school_year'] < conduct_row['school_year']) | ((student_conduct['school_year'] == conduct_row['school_year']) & (student_conduct['semester'] < conduct_row['semester']))]
                    if not prev.empty:
                        prev_row = prev.sort_values(['school_year', 'semester'], ascending=[False, False]).iloc[0]
                        trend = conduct_row['conduct_score'] - prev_row['conduct_score']
                    else:
                        trend = 0
                    conduct_trends.append(trend)
                    num_conduct_semesters.append(len(student_conduct))
                else:
                    latest_conduct_scores.append(65.0)
                    latest_conduct_semesters.append(1)
                    latest_conduct_years.append(2324)
                    latest_classifications.append('Fair')
                    conduct_trends.append(0)
                    num_conduct_semesters.append(0)

            self.df['avg_conduct_score'] = avg_conduct_scores
            self.df['latest_conduct_score'] = latest_conduct_scores
            self.df['latest_conduct_semester'] = latest_conduct_semesters
            self.df['latest_conduct_year'] = latest_conduct_years
            self.df['conduct_trend'] = conduct_trends
            self.df['latest_conduct_classification'] = latest_classifications
            self.df['num_conduct_semesters'] = num_conduct_semesters
            self.df['conduct_classification_encoded'] = self.data_loader.le_conduct_classification.fit_transform(self.df['latest_conduct_classification'])
            
            self.data_loader.conduct_features = [
                'avg_conduct_score', 'latest_conduct_score', 'latest_conduct_semester',
                'latest_conduct_year', 'conduct_trend', 'conduct_classification_encoded',
                'num_conduct_semesters'
            ]
            print(f"Successfully integrated conduct data for {len(self.df)} records (by semester/year)")
            print(f"Conduct features added: {self.data_loader.conduct_features}")
        except Exception as e:
            print(f"Warning: Could not integrate conduct data: {e}")
            self.data_loader.conduct_features = []

    def integrate_self_study_data(self):
        """Tích hợp dữ liệu tự học"""
        try:
            if self.data_loader.tuhoc_df is None:
                print("No self-study data available")
                return
            
            # Only take necessary columns
            tuhoc_df = self.data_loader.tuhoc_df[['Student_ID', 'year', 'semester', 'accumulated_study_hours', 'accumulated_study_minutes']]
            
            # Convert to string for merge
            tuhoc_df['Student_ID'] = tuhoc_df['Student_ID'].astype(str)
            tuhoc_df['year'] = tuhoc_df['year'].astype(str)
            tuhoc_df['semester'] = tuhoc_df['semester'].astype(str)
            self.df['Student_ID'] = self.df['Student_ID'].astype(str)
            self.df['year'] = self.df['year'].astype(str)
            self.df['semester'] = self.df['semester'].astype(str)
            
            # Aggregate total hours and minutes by student, year, semester
            tuhoc_agg = tuhoc_df.groupby(['Student_ID', 'year', 'semester']).agg({
                'accumulated_study_hours': 'sum',
                'accumulated_study_minutes': 'sum'
            }).reset_index()
            tuhoc_agg.rename(columns={
                'accumulated_study_hours': 'study_hours_this_semester',
                'accumulated_study_minutes': 'study_minutes_this_semester'
            }, inplace=True)
            
            # Merge into self.df by Student_ID, year, semester
            self.df = pd.merge(
                self.df,
                tuhoc_agg,
                left_on=['Student_ID', 'year', 'semester'],
                right_on=['Student_ID', 'year', 'semester'],
                how='left'
            )
            
            # Fill missing values with 0
            self.df['study_hours_this_semester'] = self.df['study_hours_this_semester'].fillna(0)
            self.df['study_minutes_this_semester'] = self.df['study_minutes_this_semester'].fillna(0)
            print("Đã tích hợp dữ liệu tự học vào self.df!")
        except Exception as e:
            print(f"Không thể tích hợp dữ liệu tự học: {e}")

    def create_teaching_method_features(self):
        """Create teaching method features"""
        tm_mapping = {}
        for _, row in self.data_loader.ppgd_df.iterrows():
            subject_id = str(row['Subject_ID'])
            tm_features = []
            for col in self.data_loader.ppgd_df.columns:
                if col.startswith('TM ') and pd.notna(row[col]):
                    tm_features.append(1)
                elif col.startswith('TM '):
                    tm_features.append(0)
            tm_mapping[subject_id] = tm_features
        
        # Add TM columns to df
        for i in range(8):
            col_name = f'TM_{i+1}'
            self.df[col_name] = 0
            self.data_loader.tm_columns.append(col_name)
        
        for idx, row in self.df.iterrows():
            subject_id = str(row['Subject_ID'])
            if subject_id in tm_mapping:
                tm_features = tm_mapping[subject_id]
                for i, value in enumerate(tm_features):
                    if i < 8:
                        self.df.at[idx, f'TM_{i+1}'] = value

    def create_assessment_method_features(self):
        """Create assessment method features"""
        em_mapping = {}
        for _, row in self.data_loader.ppdg_df.iterrows():
            subject_id = str(row['Subject_ID'])
            em_features = []
            for col in self.data_loader.ppdg_df.columns:
                if col.startswith('EM ') and pd.notna(row[col]):
                    em_features.append(1)
                elif col.startswith('EM '):
                    em_features.append(0)
            em_mapping[subject_id] = em_features
        
        for i in range(14):
            col_name = f'EM_{i+1}'
            self.df[col_name] = 0
            self.data_loader.em_columns.append(col_name)
        
        for idx, row in self.df.iterrows():
            subject_id = str(row['Subject_ID'])
            if subject_id in em_mapping:
                em_features = em_mapping[subject_id]
                for i, value in enumerate(em_features):
                    if i < 14:
                        self.df.at[idx, f'EM_{i+1}'] = value

    def finalize_features(self):
        """Finalize feature list and prepare X, y"""
        # Add demographic features if available
        if hasattr(self.data_loader, 'demographic_features'):
            available_demo_features = [col for col in self.data_loader.demographic_features if col in self.df.columns]
            self.data_loader.feature_names.extend(available_demo_features)
        
        # Add conduct features if available
        if hasattr(self.data_loader, 'conduct_features'):
            available_conduct_features = [col for col in self.data_loader.conduct_features if col in self.df.columns]
            self.data_loader.feature_names.extend(available_conduct_features)
        
        # Add self-study features
        self_study_features = ['study_hours_this_semester', 'study_minutes_this_semester']
        available_self_study_features = [col for col in self_study_features if col in self.df.columns]
        self.data_loader.feature_names.extend(available_self_study_features)
        
        # Add teaching and assessment method features
        available_tm_features = [col for col in self.data_loader.tm_columns if col in self.df.columns]
        available_em_features = [col for col in self.data_loader.em_columns if col in self.df.columns]
        self.data_loader.feature_names.extend(available_tm_features)
        self.data_loader.feature_names.extend(available_em_features)
        
        # Filter features that exist in the dataframe
        available_features = [col for col in self.data_loader.feature_names if col in self.df.columns]
        missing_features = [col for col in self.data_loader.feature_names if col not in self.df.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        # Prepare X and y
        self.X = self.df[available_features]
        self.y = self.df['passed']
        
        # Update feature names to only include available ones
        self.data_loader.feature_names = available_features 