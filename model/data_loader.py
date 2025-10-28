import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import DATA_FILES, SUBJECT_REPLACE
from .utils import convert_to_numeric, convert_to_scale_6

class DataLoader:
    def __init__(self):
        self.df = None
        self.ppgd_df = None
        self.ppdg_df = None
        self.nhankhau_df = None
        self.conduct_df = None
        self.tuhoc_df = None
        
        # Label encoders
        self.le_student_id = LabelEncoder()
        self.le_lecturer = LabelEncoder()
        self.le_subject = LabelEncoder()
        self.le_gender = LabelEncoder()
        self.le_religion = LabelEncoder()
        self.le_birth_place = LabelEncoder()
        self.le_ethnicity = LabelEncoder()
        self.le_conduct_classification = LabelEncoder()
        
        # Demographic columns
        self.gender_col = None
        self.religion_col = None
        self.birth_place_col = None
        self.ethnicity_col = None
        
        # Feature lists
        self.feature_names = None
        self.demographic_features = []
        self.conduct_features = []
        self.tm_columns = []
        self.em_columns = []
        
        # Valid subjects
        self.valid_subjects = set()

    def load_main_data(self):
        """Load main data from Excel file"""
        print("Reading data from Excel file...")
        self.df = pd.read_excel(DATA_FILES['main_data'])
        
        # Load teaching and assessment method data
        self.ppgd_df = pd.read_excel(DATA_FILES['teaching_methods'])
        self.ppdg_df = pd.read_excel(DATA_FILES['assessment_methods'])
        
        # Apply subject replacements
        for df in [self.df, self.ppgd_df, self.ppdg_df]:
            df['Subject_ID'] = df['Subject_ID'].replace(SUBJECT_REPLACE)
        
        # Get valid subjects
        ppgd_subjects = set(self.ppgd_df['Subject_ID'].dropna().astype(str))
        ppdg_subjects = set(self.ppdg_df['Subject_ID'].dropna().astype(str))
        self.valid_subjects = ppgd_subjects.intersection(ppdg_subjects)
        
        # Handle special case: INF0263 equivalent to INF0153 and INF0263
        if 'INF0263' in self.valid_subjects:
            self.valid_subjects.add('INF0153')
        
        # Filter data to only include valid subjects
        self.df = self.df[self.df['Subject_ID'].astype(str).isin(self.valid_subjects)]
        
        # Map INF0153 to INF0263
        self.df.loc[self.df['Subject_ID'] == 'INF0153', 'Subject_ID'] = 'INF0263'
        
        print(f"Số môn hợp lệ: {len(self.valid_subjects)}")

    def load_demographic_data(self):
        """Load demographic data"""
        print("Reading demographic data from nhankhau.xlsx...")
        try:
            self.nhankhau_df = pd.read_excel(DATA_FILES['demographic'])
            print(f"Successfully loaded demographic data with {len(self.nhankhau_df)} students")
            
            # Normalize Student_ID column in demographic file
            student_id_cols = [col for col in self.nhankhau_df.columns if 'student' in col.lower() or 'id' in col.lower()]
            if student_id_cols:
                self.nhankhau_df['Student_ID'] = self.nhankhau_df[student_id_cols[0]]
            
            # Find important demographic columns
            gender_cols = [col for col in self.nhankhau_df.columns if 'giới' in col.lower() or 'gender' in col.lower()]
            religion_cols = [col for col in self.nhankhau_df.columns if 'tôn' in col.lower() or 'religion' in col.lower()]
            birth_place_cols = [col for col in self.nhankhau_df.columns if 'nơi sinh' in col.lower() or 'birth' in col.lower()]
            ethnicity_cols = [col for col in self.nhankhau_df.columns if 'dân tộc' in col.lower() or 'ethnic' in col.lower()]
            
            # Select first column if found
            self.gender_col = gender_cols[0] if gender_cols else None
            self.religion_col = religion_cols[0] if religion_cols else None
            self.birth_place_col = birth_place_cols[0] if birth_place_cols else None
            self.ethnicity_col = ethnicity_cols[0] if ethnicity_cols else None
            
            print(f"Found demographic columns: Gender={self.gender_col}, Religion={self.religion_col}, Birth Place={self.birth_place_col}, Ethnicity={self.ethnicity_col}")
            
        except Exception as e:
            print(f"Warning: Could not load demographic data: {e}")
            self.nhankhau_df = None
            self.gender_col = None
            self.religion_col = None
            self.birth_place_col = None
            self.ethnicity_col = None

    def load_conduct_data(self):
        """Load conduct score data"""
        print("Integrating conduct score data...")
        try:
            self.conduct_df = pd.read_excel(DATA_FILES['conduct'])
            print(f"Successfully loaded conduct data with {len(self.conduct_df)} records")
        except Exception as e:
            print(f"Warning: Could not load conduct data: {e}")
            self.conduct_df = None

    def load_self_study_data(self):
        """Load self-study data"""
        try:
            self.tuhoc_df = pd.read_excel(DATA_FILES['self_study'])
            print("Successfully loaded self-study data")
        except Exception as e:
            print(f"Warning: Could not load self-study data: {e}")
            self.tuhoc_df = None

    def process_main_data(self):
        """Process main data and calculate scores"""
        print("Processing and preparing data...")
        
        # Process exam scores and calculate CLO
        self.df['exam_score_10'] = self.df['exam_score'].apply(convert_to_numeric)
        self.df['exam_score_6'] = self.df['exam_score_10'].apply(convert_to_scale_6)
        
        # Process final scores
        self.df['summary_score_numeric'] = self.df['summary_score'].apply(convert_to_numeric)
        
        # Mark absent cases
        self.df['is_absent_exam'] = (self.df['exam_score'].astype(str).str.upper() == 'VT') | (self.df['exam_score_10'] == 0)
        self.df['is_absent_summary'] = (self.df['summary_score'].astype(str).str.upper() == 'VT') | (self.df['summary_score_numeric'] == 0)
        
        # Calculate course result (scale 10) and CLO (scale 6)
        self.df['passed'] = ((self.df['summary_score_numeric'] >= 4) & 
                            (~self.df['is_absent_summary'])).astype(int)
        self.df['clo_achieved'] = ((self.df['exam_score_6'] >= 3.5) & 
                            (~self.df['is_absent_exam'])).astype(int)
        
        # Encode categorical variables
        self.df['student_id_encoded'] = self.le_student_id.fit_transform(self.df['Student_ID'])
        self.df['lecturer_encoded'] = self.le_lecturer.fit_transform(self.df['Lecturer_Name'])
        self.df['subject_encoded'] = self.le_subject.fit_transform(self.df['Subject_ID'])
        
        # Initialize feature names
        self.feature_names = ['student_id_encoded', 'lecturer_encoded', 'subject_encoded']
        
        print(f"Total records: {len(self.df)}")
        print(f"Number of students who passed: {sum(self.df['passed'] == 1)}")
        print(f"Number of students who failed: {sum(self.df['passed'] == 0)}")
        print(f"Number of students who achieved CLO: {sum(self.df['clo_achieved'] == 1)}")
        print(f"Number of students who did not achieve CLO: {sum(self.df['clo_achieved'] == 0)}")

    def get_available_options(self):
        """Get available options for input validation"""
        return {
            'student_id_list': sorted(self.df['Student_ID'].dropna().unique().tolist()),
            'lecturer_list': sorted(self.df['Lecturer_Name'].dropna().unique().tolist()),
            'subject_list': sorted(self.df['Subject_ID'].dropna().unique().tolist())
        }

    def validate_input(self, input_type, value):
        """Validate input against available options"""
        options = self.get_available_options()
        
        # Map input_type to correct key
        key_mapping = {
            'student_id': 'student_id_list',
            'lecturer': 'lecturer_list', 
            'subject_id': 'subject_list'
        }
        
        key = key_mapping.get(input_type, f'{input_type}_list')
        valid_list = [str(v) for v in options[key]]
        
        if str(value) not in valid_list:
            # Đặc biệt xử lý cho giảng viên mới
            if input_type == 'lecturer':
                print(f"\n⚠️ Giảng viên mới: {value}")
                print("✅ Cho phép giảng viên mới - sẽ được phân tích đặc biệt")
                return True  # Cho phép giảng viên mới
            
            print(f"\n⚠️ Error: {input_type} not found: {value}")
            print("Suggestions: Please choose one of the following values:")
            if input_type == 'student_id':
                print(", ".join(valid_list[:10]))
            else:
                print(", ".join(valid_list[:5]))
            return False
        return True

    def display_available_options(self):
        """Display available options for user input"""
        options = self.get_available_options()
        print("\nList of available Student IDs (5 samples):")
        print(", ".join(map(str, options['student_id_list'][:5])))
        print("\nList of available Lecturers (5 samples):")
        print(", ".join(options['lecturer_list'][:5]))
        print("\nList of available Subject IDs (5 samples):")
        print(", ".join(options['subject_list'][:5])) 